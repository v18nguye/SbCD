import math, copy
import json
import os
import numpy as np
from p_tqdm import p_map, t_map
import pandas as pd
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader


from collections import defaultdict
from typing import Any, Dict, List
import hydra
import omegaconf
import pytorch_lightning as pl
from torch_scatter import scatter
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_dense_batch
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from collections import Counter
from torch.distributions.poisson import Poisson
from torch.distributions.exponential import Exponential

from pyxtal.symmetry import search_cloest_wp, Group

from sbcd.common.utils import PROJECT_ROOT
from sbcd.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume, lattice_params_to_matrix_torch,
    frac_to_cart_coords, min_distance_sqr_pbc, lattice_ks_to_matrix_torch,
    sg_to_ks_mask, mask_ks, get_spacegroup_binary_repr_all, sg_to_cs, SS_DICT)
from sbcd.pl_modules.diff_utils import d_log_p_wrapped_normal
from sbcd.pl_modules.model import build_mlp
from sbcd.pl_modules.jump_utils import get_forward_rate
from scripts.compute_metrics import GenEval, get_gt_crys_ori
from scripts.crystal import Crystal
from scripts.eval_utils import lattices_to_params_shape,  get_crystals_list
from sbcd.pl_modules.lattice.crystal_family import CrystalFamily
from sbcd.pl_data.dataset import get_node_dist_marginal_asym_units

USE_THEORI_SS_MASK = False
MAX_ATOMIC_NUM=94
NUM_WYCKOFF = 186
NUM_SPACEGROUPS = 231
NUM_SYSTEMS=7
SITE_SYMM_AXES = 15
SITE_SYMM_PGS = 13
SITE_SYMM_DIM = SITE_SYMM_AXES * SITE_SYMM_PGS
SG_CONDITION_DIM = 397
SG_SYM = {spacegroup: Group(spacegroup) for spacegroup in range(1, 231)}

SG_TO_SITE_SYMM_TO_WP = dict()
SITE_SYMM_REPR = {}
for sg in range(1, 231):
    SG_TO_SITE_SYMM_TO_WP[sg] = dict()
    for wp in SG_SYM[sg].Wyckoff_positions:
        wp.get_site_symmetry()
        if wp.site_symm not in SG_TO_SITE_SYMM_TO_WP[sg]:
            SG_TO_SITE_SYMM_TO_WP[sg][wp.site_symm] = [wp]
        else:
            SG_TO_SITE_SYMM_TO_WP[sg][wp.site_symm].append(wp)
        
        if wp.site_symm not in SITE_SYMM_REPR:
            SITE_SYMM_REPR[wp.site_symm] = wp.get_site_symmetry_object().to_one_hot()

def compute_theori_prior_mask():
    SG_SYM = {spacegroup: Group(spacegroup) for spacegroup in range(1, 231)}
    site_symm_sums_per_sg = torch.zeros((231, len(SS_DICT)))
    for sg in range(1, 231):
        for wp in SG_SYM[sg].Wyckoff_positions:
            wp.get_site_symmetry()
            idx = SS_DICT[wp.site_symm]
            site_symm_sums_per_sg[sg][idx] +=1
    prior_mask = (site_symm_sums_per_sg > 0) * 1.
    return prior_mask

class DiscreteNoise(nn.Module):
    ''' priors per shared crystal system.
    '''
    def __init__(self, atom_type_prior, site_symm_prior, beta_scheduler, P_ss, P_a, use_atom_per_sg=True):
        super().__init__()
        self.use_atom_per_sg = use_atom_per_sg
        self.beta_scheduler = beta_scheduler
        self.site_symm_prior = site_symm_prior
        self.site_symm_prior_mask = compute_theori_prior_mask() # computed theorically, not emperically
        self.atom_type_prior = atom_type_prior
        self.P_ss = P_ss 
        self.P_a = P_a
        self.site_symm_pgs = SITE_SYMM_PGS
        self.site_symm_axes = SITE_SYMM_AXES
        self.max_atomic_num = MAX_ATOMIC_NUM

    #* ss means site symmetry
    def ss_to_sections(self, ss):
        return [ss[..., i*self.site_symm_pgs:(i+1)*self.site_symm_pgs] for i in range(self.site_symm_axes)]

    def reshape_ss(self, ss):
        return ss.reshape(-1, SITE_SYMM_AXES, self.site_symm_pgs) 

    def q_t(self, P, t):
        alpha = self.beta_scheduler.alphas[t]
        num_classes = P.shape[-1]
        return alpha.view(-1,1,1)*torch.eye(num_classes, device=P.device)+(1-alpha.view(-1,1,1))*P
    
    def q_t_bar(self, P, t):
        alpha_bar = self.beta_scheduler.alphas_cumprod[t]
        num_classes = P.shape[-1]
        return alpha_bar.view(-1,1,1)*torch.eye(num_classes, device=P.device)+ (1 - alpha_bar.view(-1,1,1))*P

    def q_s2t_bar(self, Ps, s, Pt, t):
        '''from s to t
        data <  s < t < noise
        s: SB event time

        @ Ps, Pt: B, K, K
        @ t,s: B
        '''
        K = Ps.shape[-1]
        ps = Ps[:,0,:] # extract the marignal prob for all samples
        pt = Pt[:,0,:]

        alpha_bar_s = self.beta_scheduler.alphas_cumprod[s]
        alpha_bar_t = self.beta_scheduler.alphas_cumprod[t]

        denom = (alpha_bar_s*(1-alpha_bar_t)).unsqueeze(-1)
        pi_t = ps*(alpha_bar_t*(1-alpha_bar_s)).unsqueeze(-1) + pt*(alpha_bar_s-alpha_bar_t).unsqueeze(-1)
        pi_t = pi_t/denom
        Pi_t = pi_t.unsqueeze(1).expand(-1,K, -1)
        assert (Pi_t[:,0,:]==Pi_t[:,-1,:]).all() # check valid Pi for all samples

        return alpha_bar_t.view(-1,1,1)*torch.eye(K, device=Ps.device)+ (1 - alpha_bar_t.view(-1,1,1))*Pi_t
    
    # atom w/o sym break
    def q_t_atom(self, t):
        return self.q_t(self.P_a, t)

    def q_t_bar_atom(self, t):
        return self.q_t_bar(self.P_a, t)
    
    def apply_atom_noise(self, atom_type, t):
        Q_t_bar = self.q_t_bar_atom(t)
        prob_atom_types = atom_type @  Q_t_bar
        return prob_atom_types
    
    # atom with sym break
    def q_t_atom_sb(self, t, sg):
        return self.q_t(self.P_a[sg], t)

    def q_t_bar_atom_sb(self, t, sg):
        return self.q_t_bar(self.P_a[sg], t)
    
    def q_s2t_bar_atom_sb(self, t, s, sg_t, sg_s):
        return self.q_s2t_bar(self.P_a[sg_s], s, self.P_a[sg_t], t)

    def apply_atom_noise_sb(self, atom_type, t, sg_t, sg_0, s):
        Q_t_bar = self.q_s2t_bar_atom_sb(t, s, sg_t, sg_0)
        prob_atom_types = atom_type @  Q_t_bar
        return prob_atom_types

    # site syms
    def q_t_ss_sb(self, t, sg):
        return self.q_t(self.P_ss[sg], t)

    def q_t_bar_ss_sb(self, t, sg):
        return self.q_t_bar(self.P_ss[sg], t)
    
    def q_s2t_bar_ss_sb(self, t, s, sg_t, sg_s):
        '''ss symmetry breaking
        data < s < t < noise
        '''
        return self.q_s2t_bar(self.P_ss[sg_s], s, self.P_ss[sg_t], t)
    
    def apply_site_symm_noise_sb(self, site_symm, t, sg_t, sg_0, s):
        Q_t_bar = self.q_s2t_bar_ss_sb(t, s, sg_t, sg_0)
        prob_site_symms = site_symm @  Q_t_bar
        return prob_site_symms
    
    def sigma_sqr_ratio(self, s_int, t_int):
        return self.beta_scheduler.alphas_cumprod[t_int] / self.beta_scheduler.alphas_cumprod[s_int]
    
    def sample_limit_dist(self, node_mask, use_sg_prior):
        """ Sample from the limit distribution of the diffusion process. 
        Start with the 1st spacegroup.
        """
        bs, n_max = node_mask.shape
        
        U_sg = torch.ones(bs).long() # start with the 1st sg of triclinic.
        
        if self.use_atom_per_sg:
            a_limit = self.atom_type_prior[U_sg].unsqueeze(1).expand(-1, n_max, -1)
        else:
            a_limit = self.atom_type_prior.expand(bs, n_max, -1)

        U_a = a_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_a = F.one_hot(U_a, num_classes=a_limit.shape[-1]).float()
        U_a = U_a * node_mask.unsqueeze(-1)

        # ss_limit = self.site_symm_prior[U_sg].unsqueeze(-2).expand(bs, n_max, -1)
        ss_limit = self.site_symm_prior[U_sg].unsqueeze(1).expand(-1, n_max, -1)
        U_ss = ss_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max).to(node_mask.device)
        U_ss = F.one_hot(U_ss, num_classes=ss_limit.shape[-1]).float()
        U_ss = U_ss * node_mask.unsqueeze(-1)
        
        return U_a, U_ss, U_sg.to(node_mask.device)

    def sample_discrete_features(self, prob_a, prob_ss, node_mask, sg=None):
        ''' Sample features from multinomial distribution with given probabilities (probX, probE, proby)
            :param prob_a: bs, n, dx_out        node features
            :param prob_ss: bs, n, dx_out        node features
        '''
        bs, n = node_mask.shape
        # The masked rows should define probability distributions as well (uniform over 94 species)
        prob_a[~node_mask] = 1 / prob_a.shape[-1]
        prob_ss[~node_mask] = 1/prob_ss.shape[-1]
        # Flatten the probability tensor to sample with multinomial
        prob_a = prob_a.reshape(bs * n, -1)       # (bs * n, dx_out)
        # Sample a
        atom_t = prob_a.multinomial(1)                                  # sampling axis=1, (bs * n, 1)
        atom_t = atom_t.reshape(bs, n)     # (bs, n)
        atom_t = F.one_hot(atom_t, num_classes=prob_a.shape[-1]).float()
        # Sample ss
        prob_ss = prob_ss.reshape(bs * n, -1)
        if sg is not None:
            # manifold preserving proj
            if USE_THEORI_SS_MASK:
                sg_mask = ((self.site_symm_prior_mask.to(prob_ss.device))[sg] > 0.)
            else:
                sg_mask = ((self.site_symm_prior.to(prob_ss.device))[sg] > 0.)
            sg_mask = sg_mask.unsqueeze(1).expand(-1,n,-1)
            sg_mask = sg_mask.reshape(bs*n, -1)
            # masking
            prob_ss = prob_ss*sg_mask
            # normalized
            prob_ss = prob_ss / torch.sum(prob_ss, dim=-1, keepdim=True)
        site_symm_t = prob_ss.multinomial(1)
        site_symm_t = site_symm_t.reshape(bs, n)
        site_symm_t = F.one_hot(site_symm_t, num_classes=len(SS_DICT)).float()

        return atom_t, site_symm_t


    def p_s_and_t_given_0(self, z_t, Qt, Qsb, Qtb): # b=bar
        """ M: X, E or charges
            Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
            X_t: bs, n, dt
            Qt: bs, d_t-1, dt
            Qsb: bs, d0, d_t-1
            Qtb: bs, d0, dt.
        """
        Qt_T = Qt.transpose(-1, -2)                 # bs, dt, d_t-1
        left_term = z_t @ Qt_T                      # bs, N, d_t-1
        left_term = left_term.unsqueeze(dim=2)      # bs, N, 1, d_t-1

        right_term = Qsb.unsqueeze(1)               # bs, 1, d0, d_t-1
        numerator = left_term * right_term          # bs, N, d0, d_t-1

        X_t_transposed = z_t.transpose(-1, -2)      # bs, dt, N

        prod = Qtb @ X_t_transposed                 # bs, d0, N
        prod = prod.transpose(-1, -2)               # bs, N, d0
        denominator = prod.unsqueeze(-1)            # bs, N, d0, 
        denominator[denominator == 0] = 1e-6

        out = numerator / denominator
        return out
    
    def p_s_and_t_given_0_a(self, z_t_a, t, s):
        Qtb_a = self.q_t_bar_atom(t)
        Qsb_a = self.q_t_bar_atom(s)
        Qt_a = self.q_t_atom(t)
        return self.p_s_and_t_given_0(z_t_a,Qt_a, Qsb_a, Qtb_a)
    
    def p_s_and_t_given_0_a_sb(self, z_t_a, t, s, sg_t, sg_s):
        Qtb_a = self.q_s2t_bar_atom_sb(t, s, sg_t, sg_s)
        Qsb_a = self.q_t_bar_atom_sb(s, sg_s)
        Qt_a = self.q_t_atom_sb(t, sg_t)
        return self.p_s_and_t_given_0(z_t_a,Qt_a, Qsb_a, Qtb_a)

    def p_s_and_t_given_0_ss_sb(self, z_t_ss, t, s, sg_t, sg_s):
        Qtb_ss = self.q_s2t_bar_ss_sb(t, s, sg_t, sg_s)
        Qsb_ss = self.q_t_bar_ss_sb(s, sg_s)
        Qt_ss = self.q_t_ss_sb(t, sg_t)
        return self.p_s_and_t_given_0(z_t_ss,Qt_ss, Qsb_ss, Qtb_ss)

    def sample_zs_from_zt_and_pred(self, z_t_a, z_t_ss, pred_a, pred_ss, t, s, node_mask, sg_t, sg_s):
        """Samples from zs ~ p(zs | zt). Only used during sampling. 
        data < s < t < noise
        """
        # Normalize predictions for the categorical features
        # z_t_ss += 1e-5 # noise perbutation to help 'prob_ss' to be projected back to the marginal prior manifold
        p_s_and_t_given_0_site_symms = self.p_s_and_t_given_0_ss_sb(z_t_ss, t, s, sg_t, sg_s)

        if self.use_atom_per_sg:
            p_s_and_t_given_0_atom_types = self.p_s_and_t_given_0_a_sb(z_t_a, t, s, sg_t, sg_s)
        else:
            p_s_and_t_given_0_atom_types = self.p_s_and_t_given_0_a(z_t_a, t, s)

        # Dim of these two tensors: bs, N, d0, d_t-1
        #pred_a = F.softmax(pred_a, dim=-1)               # bs, n, d0
        weighted_a = pred_a.unsqueeze(-1) * p_s_and_t_given_0_atom_types         # bs, n, d0, d_t-1
        unnormalized_prob_a = weighted_a.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_a[torch.sum(unnormalized_prob_a, dim=-1) == 0] = 1e-5
        prob_a = unnormalized_prob_a / torch.sum(unnormalized_prob_a, dim=-1, keepdim=True)  # bs, n, d_t-1

        weighted_ss = pred_ss.unsqueeze(-1) * p_s_and_t_given_0_site_symms         # bs, n, d0, d_t-1
        unnormalized_prob_ss = weighted_ss.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_ss[torch.sum(unnormalized_prob_ss, dim=-1) == 0] = 1e-5
        prob_ss = unnormalized_prob_ss / torch.sum(unnormalized_prob_ss, dim=-1, keepdim=True)  # bs, n, d_t-1

        assert ((prob_a.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_ss.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_a_s, sampled_ss_s = self.sample_discrete_features(prob_a, prob_ss, node_mask, sg_s)
        return sampled_a_s, sampled_ss_s

    def discrete_loss(self, sample_a, sample_ss, pred_a, pred_ss):
        '''
        Cross entropy loss for atom_types as well as each site_symm component
        '''
        loss_a = F.nll_loss(torch.log(pred_a + 1e-20), sample_a)
        loss_ss =  F.nll_loss(torch.log(pred_ss + 1e-20), sample_ss)
        return loss_a, loss_ss

class DiscreteNoiseMarginal(DiscreteNoise):
    def __init__(self, use_atom_per_sg, atom_marginals_path, ss_marginals_path, beta_scheduler):

        atom_type_prior = torch.load(atom_marginals_path) # 94
        site_symm_prior_per_sg = torch.load(ss_marginals_path) # 231x81
        # 231x81x81
        P_ss = nn.Parameter(site_symm_prior_per_sg.unsqueeze(1).expand(-1, len(SS_DICT), -1).clone(), requires_grad=False)
        if use_atom_per_sg:
            P_a = nn.Parameter(atom_type_prior.unsqueeze(1).expand(-1, MAX_ATOMIC_NUM, -1).clone(), requires_grad=False)
        else:
            P_a = nn.Parameter(atom_type_prior.unsqueeze(0).expand(MAX_ATOMIC_NUM, -1).clone(), requires_grad=False)
        super().__init__(atom_type_prior, site_symm_prior_per_sg, beta_scheduler, P_ss, P_a, use_atom_per_sg)

def find_num_atoms(dummy_ind, total_num_atoms):
    # num_atoms states how many atoms are there in each crystal (num_repr + dummy origin)
    actual_num_atoms = []
    atoms = 0
    for num in total_num_atoms:
        # find number of 0 in dummy_ind from atoms to atoms+num
        actual_num_atoms.append(torch.sum(dummy_ind[atoms:atoms+num] == 0).item())
        atoms += num
        
    return torch.tensor(actual_num_atoms)

def split_argmax_sitesymm(site_symm:torch.Tensor) -> np.ndarray:
    # site_symm : num_repr x 66
    return np.array(np.abs(1 - site_symm.cpu().detach().numpy()) < 0.1, dtype=float)


def modify_frac_coords_one(frac_coords, site_symm, atom_types, sg):
    '''wp links
    https://onlinelibrary.wiley.com/iucr/itc/Ac/contents/
    '''
    sg = sg.item()
    site_symm_axis = site_symm.argmax(-1).detach().cpu()

    # Get site symmetry of WP for the spacegroup
    wp_to_site_symm = SG_TO_SITE_SYMM_TO_WP[sg]
    order_all_ss_list = [ss for ss,_ in SS_DICT.items()]

    # iterate over frac coords and corresponding site-symm
    new_frac_coords, new_atom_types, new_site_symm = [], [], []
    min_ss_dists = []
    wp_projection_dists = []
    for (sym_num, frac_coord, atm_type) in zip(site_symm_axis, frac_coords, atom_types):
        frac_coord = frac_coord.cpu().detach().numpy()

        # Get all WPs that are closest in terms of site symmetry (generated)
        ss_name = order_all_ss_list[sym_num]
        closes = []
        if ss_name in wp_to_site_symm:
            closest_ss_wps = wp_to_site_symm[ss_name]
            # For each WP find closest position in space
            for wp in closest_ss_wps:
                for orbit_index in range(len(wp.ops)):
                    # find the closest pos wrt the current ss (wp.ops)
                    close = search_cloest_wp(SG_SYM[sg], wp, wp.ops[orbit_index], frac_coord)%1.
                    closes.append((close, wp, orbit_index, np.linalg.norm(np.minimum((close - frac_coord)%1., (frac_coord - close)%1.))))
        else:
            raise ValueError(f'Cant not find HM symbol {ss_name} in space group {sg}')

        try:
            # pick the nearest wp to project
            closest_rank_list = sorted(closes, key=lambda x: x[-1])
            dist_list = [x[-1] for x in closest_rank_list]
            closest = sorted(closes, key=lambda x: x[-1])[0]
            wyckoff = closest[1]
            repr_index = closest[2]
            wp_projection_dists.append(closest[3])
            # use wp operations on frac_coord assymetric unit to create the unit cell
            frac_coord = closest[0]
            for index in range(len(wyckoff)):
                # new_frac_coords.append(wyckoff[index].operate(frac_coord)%1.)
                new_frac_coords.append(wyckoff[(index + repr_index) % len(wyckoff)].operate(frac_coord)%1.)
                new_atom_types.append(atm_type.cpu().detach().numpy())
                new_site_symm.append(ss_name)
            
            # check all closest ss having the same wyckoff
            if len(closest_ss_wps) > 1:
                dist_inds = list(np.where(dist_list == closest[-1])[0])
                for i in dist_inds:
                    assert closest_rank_list[i][1] == wyckoff
                    
        except:
            # print('Weird things happen, and I did not predict correctly')
            new_frac_coords.append(frac_coord)
            new_atom_types.append(atm_type.cpu().detach().numpy())
            new_site_symm.append(ss_name)

    new_frac_coords = np.stack(new_frac_coords)
    new_atom_types = np.stack(new_atom_types)
    # new_site_symm = np.stack(new_site_symm)
    return new_frac_coords, len(new_frac_coords), new_atom_types, new_site_symm, min_ss_dists, wp_projection_dists


def modify_frac_coords(traj:Dict, spacegroups:List[int], num_repr:List[int]) -> Dict:
    device = traj['frac_coords'].device
    total_atoms = 0
    updated_frac_coords = []
    updated_num_atoms = []
    updated_atom_types = []
    updated_site_symm = []
    min_ss_dists, wp_projection_dists = [], []
   
    for index in range(len(num_repr)):
        if num_repr[index] > 0:
            new_frac_coords, new_num_atoms, new_atom_types, new_site_sym,  min_ss_dist, wp_projection_dist = modify_frac_coords_one(
                    traj['frac_coords'][total_atoms:total_atoms+num_repr[index]],
                    traj['site_symm'][total_atoms:total_atoms+num_repr[index]],
                    traj['atom_types'][total_atoms:total_atoms+num_repr[index]],
                    spacegroups[index], 
                )

            if new_num_atoms:
                updated_frac_coords.append(new_frac_coords)
                updated_num_atoms.append(new_num_atoms)
                updated_atom_types.append(new_atom_types)
                updated_site_symm.append(new_site_sym)
                min_ss_dists.append(min_ss_dist)
                wp_projection_dists.append(wp_projection_dist)
        total_atoms += num_repr[index]
    
    traj['frac_coords'] = torch.cat([torch.from_numpy(x) for x in updated_frac_coords]).to(device)
    traj['atom_types'] = torch.cat([torch.from_numpy(x) for x in updated_atom_types]).to(device)
    traj['num_atoms'] = torch.tensor(updated_num_atoms).to(device)
    # traj['site_symm'] = torch.cat([torch.from_numpy(x) for x in updated_site_symm]).to(device)
    traj['min_ss_dists'] = min_ss_dists
    traj['wp_projection_dists'] = wp_projection_dists

    return traj

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()

    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 10,
            "monitor": "val_loss",
            "strict": True,
            "name": None,
        }
        return {"optimizer": opt, "lr_scheduler": lr_scheduler_config}


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TimeCrystal(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ratio_eps = 1e-9
        self.use_atom_per_sg = self.hparams.data.use_atom_per_sg # use marginal atom type dist per sg
        self.one_forward_pass = self.hparams.data.one_forward_pass
        assert self.hparams.prior !='masked' # legacy code check
        self.decoder = hydra.utils.instantiate(self.hparams.decoder, time_dim=self.hparams.time_dim + self.hparams.latent_dim, latent_dim = self.hparams.latent_dim, pred_type = True, pred_site_symm_type = True, smooth = True, max_atoms=MAX_ATOMIC_NUM)
        self.beta_scheduler = hydra.utils.instantiate(self.hparams.beta_scheduler).to(self.device)
        self.sigma_scheduler = hydra.utils.instantiate(self.hparams.sigma_scheduler).to(self.device)
        self.time_dim = self.hparams.time_dim
        self.latent_dim = self.hparams.latent_dim
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.sg_embedding = build_mlp(in_dim=NUM_SPACEGROUPS, hidden_dim=128, fc_num_layers=2, out_dim=self.latent_dim)
        self.discrete_noise = self.init_discrete_noise(self.hparams.prior)
        self.crystal_family = CrystalFamily()
        self.node_dist = get_node_dist_marginal_asym_units(self.hparams.data.datamodule.datasets.train.save_path)
        self.jump_process = get_forward_rate(self.hparams.data.rate_function_name, self.hparams.data, device='cuda:0')

    def on_train_start(self):
        log_dict = {
            'comp_valid': 0,
            'struct_valid': 0,
            'valid': 0,
            'wdist': 5, 
        }
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def ks_noise_sb(self, ks_0, sg_0, sg_t, t, s):
        '''lattice noise crystal system constraint symmetry breaking
        '''
        alphas_cumprod_t = self.beta_scheduler.alphas_cumprod[t]
        c0 = torch.sqrt(alphas_cumprod_t)
        c1 = torch.sqrt(1. - alphas_cumprod_t)

        alphas_cumprod_s = self.beta_scheduler.alphas_cumprod[s]
        c3 = alphas_cumprod_t/alphas_cumprod_s

        rand_ks = torch.randn_like(ks_0)
        ks_t = c0[:, None] * ks_0 + c1[:, None] * rand_ks
        ks_t = self.crystal_family.proj_k_to_spacegroup(ks_t, sg_0)
        ks_t -= self.crystal_family.proj_k_to_spacegroup(torch.sqrt(1-c3[:,None])*rand_ks, sg_0)

        rand_ks = torch.randn_like(ks_0)
        ks_t += self.crystal_family.proj_k_to_spacegroup(torch.sqrt(1-c3[:,None])*rand_ks, sg_t)
        ks_t += self.crystal_family.get_sg_mask(sg_t)*self.crystal_family.get_sg_bias(sg_0)*torch.sqrt(c3[:,None])
        return ks_t


    def init_discrete_noise(self, prior='marginal'):
        if prior == 'marginal':
            return DiscreteNoiseMarginal(self.use_atom_per_sg, self.hparams.data.datamodule.atom_marginals_path,
                                         self.hparams.data.datamodule.ss_marginals_path, 
                                        self.beta_scheduler)
        else:
            raise NotImplementedError
        
    def get_noisy_attrs(self, num_atoms, node_mask, \
                        gt_ks, gt_sg, gt_frac_coords, \
                        sg_t_onehot, gt_atom_types_onehot, gt_site_symm_binary,\
                        sg_t, t, s):
        
        sigmas = self.sigma_scheduler.sigmas[t]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[t]
        
        input_ks = self.ks_noise_sb(gt_ks, gt_sg, sg_t, t, s)
        input_lattice = self.crystal_family.v2m(input_ks)
        lattice_feats = input_ks
            
        rand_x = torch.randn_like(gt_frac_coords)
        sigmas_per_atom = sigmas.repeat_interleave(num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(num_atoms)[:, None]

        input_frac_coords = (gt_frac_coords + sigmas_per_atom * rand_x) % 1.

        if self.use_atom_per_sg:
            atom_type_t_probs = self.discrete_noise.apply_atom_noise_sb(gt_atom_types_onehot, t, sg_t, gt_sg, s)
        else:
            atom_type_t_probs = self.discrete_noise.apply_atom_noise(gt_atom_types_onehot, t)

        site_symm_t_probs = self.discrete_noise.apply_site_symm_noise_sb(gt_site_symm_binary, t, sg_t, gt_sg, s)
        atom_types_t_onehot, site_symms_t_onehot = self.discrete_noise.sample_discrete_features(atom_type_t_probs, site_symm_t_probs, node_mask)
        symm_t = site_symms_t_onehot[node_mask] # (, 195)
        atom_types_t = atom_types_t_onehot[node_mask] # (, 94)

        sg_t_emb = self.sg_embedding(sg_t_onehot)
        time_emb = torch.cat([self.time_embedding(t), sg_t_emb], dim=-1)

        return rand_x, input_lattice, lattice_feats, input_frac_coords, symm_t, atom_types_t, time_emb, sigmas_norm_per_atom, sigmas_per_atom


    def forward(self, batch):
        batch_size = batch.num_graphs

        # diffusion time
        ti = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        ti = torch.clamp(ti, max=self.beta_scheduler.timesteps-1, min=0)  # (0,999)
        ti_norm = ti/self.beta_scheduler.timesteps

        # holding-time posterior
        tau = (self.jump_process.sampling_holding_time(ti_norm)*self.beta_scheduler.timesteps).long()
        tau = torch.clamp(tau, max=self.beta_scheduler.timesteps-1, min=0) # (0,999)

        ## jump process on sg transitions
        gt_sg = batch.spacegroup # 1st -> P1
        
        pt0_trans = self.jump_process.transition(ti_norm)
        qt_rate = self.jump_process.rate(ti_norm)

        sg_ti = pt0_trans[torch.arange(batch_size, device=self.device), gt_sg, :] # B, S
        sg_ti = Categorical(sg_ti).sample()

        # time ops and masking
        sg_mask = (sg_ti==gt_sg)
        wi = (0*sg_mask +(1-sg_mask*1.)*torch.clamp(ti-tau, min=1)).long()
        ts = torch.clamp(ti-wi, min=0)

        t_next = (ti+1).long()
        w_next = (1*sg_mask +(1-sg_mask*1.)*torch.clamp(t_next-tau, min=1)).long()
        ts_next = torch.clamp(t_next-w_next, min=0)
        
        assert sg_ti[sg_ti!=gt_sg].sum() == sum(sg_ti!=gt_sg) # check all transitions to P1 ONLY

        sg_tnext = torch.ones_like(sg_ti, device=self.device) # xtidle(i,j) = (1 - dirac_{ij}) Q_t(i,j) / sum_{j \ne i)(i,j)
                                                            # so totally P1 sgs

        sg_ti_onehot = F.one_hot(sg_ti, num_classes=NUM_SPACEGROUPS).float()
        sg_tnext_onehot = F.one_hot(sg_tnext, num_classes=NUM_SPACEGROUPS).float()

        ## get M0 values
        site_symm = batch.site_symm # bs, 15, 13
        gt_site_symm_binary, node_mask = to_dense_batch(site_symm, batch.batch, fill_value=0) # bs, n_max, 195

        gt_frac_coords = batch.frac_coords

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattices = self.crystal_family.de_so3(lattices)
        gt_ks = self.crystal_family.m2v(lattices)
        gt_ks = self.crystal_family.proj_k_to_spacegroup(gt_ks, gt_sg)

        atom_types, _ = to_dense_batch(batch.atom_types - 1, batch.batch, fill_value=0) # value, index
        gt_atom_types_onehot = F.one_hot(atom_types, num_classes=self.discrete_noise.max_atomic_num).float()

        # p(M_t|M_0)
        rand_x, input_lattice, lattice_feats, input_frac_coords, \
            symm_t, atom_types_t, time_emb, sigmas_norm_per_atom, sigmas_per_atom = self.get_noisy_attrs(batch.num_atoms, node_mask, \
                                                                                            gt_ks, gt_sg, gt_frac_coords, \
                                                                                            sg_ti_onehot, gt_atom_types_onehot, gt_site_symm_binary,\
                                                                                            sg_ti, ti, ts)
        
        # 1st forward pass
        pred_lattice, pred_x, pred_t_logit, pred_symm_logit, plogit_sg0_t = self.decoder(time_emb, atom_types_t, input_frac_coords, 
                                                    lattice_feats, input_lattice, batch.num_atoms, 
                                                    batch.batch, site_symm_probs=symm_t)
        
        _, _input_lattice, _lattice_feats, _input_frac_coords, \
            _symm_t, _atom_types_t, _time_emb, _, _ = self.get_noisy_attrs(batch.num_atoms, node_mask, \
                                                                            gt_ks, gt_sg, gt_frac_coords, \
                                                                            sg_tnext_onehot, gt_atom_types_onehot, gt_site_symm_binary,\
                                                                            sg_tnext, t_next, ts_next)
        
        _, _, _, _, plogit_sg0_tidle = self.decoder(_time_emb, _atom_types_t, _input_frac_coords, 
                                                        _lattice_feats, _input_lattice, batch.num_atoms, 
                                                        batch.batch, site_symm_probs=_symm_t)
        # suppervision from M_t
        pred_sg = F.softmax(plogit_sg0_t, -1)
        loss_sg =  F.nll_loss(torch.log(pred_sg + 1e-20), gt_sg)

        pred_t = F.softmax(pred_t_logit, -1) # (, 94)
        pred_symm = F.softmax(pred_symm_logit, -1) # (, 81)

        loss_type, loss_symm = self.discrete_noise.discrete_loss(batch.atom_types - 1, batch.site_symm.argmax(-1), pred_t, pred_symm)

        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom) # (, 3)
        loss_coord = F.mse_loss(pred_x, tar_x)

        # pred_lattice = self.crystal_family.proj_k_to_spacegroup(pred_lattice, gt_sg)
        loss_lattice = F.mse_loss(pred_lattice, gt_ks) 

        # first elbo term (regularization)
        mask_reg = torch.ones((batch_size,NUM_SPACEGROUPS), device=self.device) # (B,S)
        mask_reg[torch.arange(batch_size, device=self.device), sg_ti] = 0.0 # masking no self-transition involed

        pt0_numer_reg = pt0_trans.detach().clone()
        pt0_denom_reg = pt0_trans[torch.arange(batch_size, device=self.device), :, sg_ti] + self.ratio_eps # all x0 -> xt
        rate_reg = qt_rate[torch.arange(batch_size, device=self.device), :, sg_ti] # all x' -> xt

        reg_tmp = (((mask_reg * rate_reg).unsqueeze(1)) @ pt0_numer_reg.transpose(1,2)).squeeze()
        reg_term = torch.sum((pred_sg / pt0_denom_reg)*reg_tmp, dim=-1)

        # second elbo term (normalisation)
        pred_sg_tidle = F.softmax(plogit_sg0_tidle, -1) # all x0
        pt0_numer_num = pt0_trans[torch.arange(batch_size, device=self.device), :, sg_ti] # all x0 -> xt
        pt0_denom_num = pt0_trans[torch.arange(batch_size, device=self.device), :, sg_tnext] + self.ratio_eps # all x0 -> x_tidle


        inner_norm = qt_rate[torch.arange(batch_size, device=self.device), sg_ti, sg_tnext] # x -> x_tidle
        inner_norm *= (pt0_numer_num/pt0_denom_num*pred_sg_tidle).sum(-1)
        log_norm = torch.log(inner_norm + self.ratio_eps)

        zt = (mask_reg * qt_rate[torch.arange(batch_size, device=self.device), sg_ti, :]).sum(-1) # xt-> all x'

        norm_term = zt*log_norm

        loss_rate_elbo = torch.mean(reg_term - norm_term)

        loss = (
            self.hparams.data.cost_lattice * loss_lattice +
            self.hparams.data.cost_coord * loss_coord + 
            self.hparams.data.cost_type * loss_type +
            self.hparams.data.cost_symm * loss_symm +
            self.hparams.data.cost_sg * loss_sg +
            self.hparams.data.cost_rate * loss_rate_elbo
        )
        return {
            'loss' : loss,
            'loss_lattice' : loss_lattice,
            'loss_coord' : loss_coord,
            'loss_type' : loss_type,
            'loss_symm' : loss_symm,
            'loss_sg': loss_sg,
            'loss_rate': loss_rate_elbo
        }

    @torch.no_grad()
    def sample(self, batch_size, diff_ratio = 1.0, step_lr = 1e-5):
        '''
        '''
        num_atoms = self.node_dist.sample((batch_size,)).to(self.device)
        num_nodes = sum(num_atoms)
        batch = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(num_atoms)]).to(self.device)
        _, node_mask = to_dense_batch(batch, batch, fill_value=0)
        
        t_T, symm_T, sg_T = self.discrete_noise.sample_limit_dist(node_mask, use_sg_prior=False)

        k_T = torch.randn([batch_size, 6]).to(self.device)
        k_T = self.crystal_family.proj_k_to_spacegroup(k_T, sg_T)
        l_T = self.crystal_family.v2m(k_T)
    
        x_T = torch.rand([num_nodes, 3]).to(self.device)
        t_T = t_T[node_mask]
        symm_T = symm_T[node_mask]

        traj = {self.beta_scheduler.timesteps : {
            'num_atoms' : num_atoms,
            'atom_types' : t_T,
            'site_symm' : symm_T,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T,
            'ks' : k_T,
            'spacegroup': sg_T,
        }}

        for t in tqdm(range(self.beta_scheduler.timesteps, 0, -1)):
            h_step = 1/self.beta_scheduler.timesteps
            times = torch.full((batch_size, ), t, device = self.device)
            times_norm = times/self.beta_scheduler.timesteps

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]
            sigmas = self.beta_scheduler.sigmas[t]

            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]

            # c0 = 1.0 / torch.sqrt(alphas)
            # c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)
            alphas_cumprod_minus_one = self.beta_scheduler.alphas_cumprod[t-1]
            c0 = torch.sqrt(alphas_cumprod_minus_one)*(1-alphas)/(1-alphas_cumprod)
            c1 = torch.sqrt(alphas)*(1-alphas_cumprod_minus_one)/(1-alphas_cumprod)

            x_t = traj[t]['frac_coords']
            t_t = traj[t]['atom_types']
            symm_t = traj[t]['site_symm']
            l_t = traj[t]['lattices']
            k_t = traj[t]['ks']
            lattice_feats_t = k_t
            sg_t = traj[t]['spacegroup']
            sg_t_onehot = F.one_hot(sg_t, num_classes=NUM_SPACEGROUPS).float()

            sg_emb = self.sg_embedding(sg_t_onehot)
            time_emb = torch.cat([self.time_embedding(times), sg_emb], dim=-1)

            # Corrector
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            _, pred_x, _, _, _ = self.decoder(time_emb, t_t, x_t, 
                                                lattice_feats_t, l_t, num_atoms, 
                                                batch, site_symm_probs=symm_t)

            x_t_minus_05 = x_t - step_size * pred_x * torch.sqrt(sigma_norm) + std_x * rand_x

            l_t_minus_05 = l_t
            k_t_minus_05 = k_t
            lattice_feats_t_minus_05 = k_t_minus_05

            t_t_minus_05 = t_t

            symm_t_minus_05 = symm_t


            # Predictor
            rand_k = torch.randn_like(k_T) if t > 1 else torch.zeros_like(k_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x, pred_t_logit, pred_symm_logit, plogit_sg0_t = self.decoder(time_emb, t_t_minus_05, x_t_minus_05, 
                                                        lattice_feats_t_minus_05, l_t_minus_05, num_atoms, 
                                                        batch, site_symm_probs=symm_t_minus_05)
            
            # rate reverse on space group
            pred_sg0 = F.softmax(plogit_sg0_t, -1) # xt -> all x0 B, S
            pt0_trans = self.jump_process.transition(times_norm) # x0 -> xt B, S, S
            qt_rate = self.jump_process.rate(times_norm) # x_(t-1) -> xt B, S, S

            pt0_denom = pt0_trans[torch.arange(batch_size, device=self.device), :, sg_t] + self.ratio_eps # all x0 -> xt B,S
            qt0_numer = pt0_trans.detach().clone() # qt0[x0 -> x_(t-1)] B,S,S. NOTE. this is state index other than time index for x_(t-1) or s in their paper
            
            inner_sum = (((pred_sg0/pt0_denom).unsqueeze(1))@qt0_numer).squeeze() # (xt -> all x0) * (all x0 -> x_(t-1))= (xt-> all x_(t-1)) B, S
            forward_rate = qt_rate[torch.arange(batch_size, device=self.device),:, sg_t] # xt -> all x_(t-1)
            reverse_rates = forward_rate*inner_sum # (xt -> all x_(t-1))* (xt-> all x_(t-1)) => xt -> all x_(t-1), B,S
            reverse_rates[torch.arange(batch_size, device=self.device), sg_t] = 0. # self transition =0.

            diffs = torch.arange(NUM_SPACEGROUPS, device=self.device).view(1,NUM_SPACEGROUPS) - sg_t.view(batch_size,1) # B,S
            poisson_dist = Poisson(reverse_rates * h_step)
            jump_nums = poisson_dist.sample() # B,S
            adj_diffs = jump_nums * diffs
            overall_jump = torch.sum(adj_diffs, dim=1) # sum over s, -> B
            sgp = sg_t + overall_jump
            sg_t_minus_1 = torch.clamp(sgp, min=1, max=NUM_SPACEGROUPS-1).long()

            assert sg_t[sg_t_minus_1!=sg_t].sum() == sum(sg_t_minus_1!=sg_t) # check all transitions start from P1 ONLY
            # print(sg_t_minus_1)
            # print('-'*100)

            # reverse others
            pred_t = F.softmax(pred_t_logit, -1) # node levels
            pred_symm = F.softmax(pred_symm_logit, -1)

            pred_t, _ = to_dense_batch(pred_t, batch, fill_value=0) # graph levels
            pred_symm, _ = to_dense_batch(pred_symm, batch, fill_value=0)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x * torch.sqrt(sigma_norm) + std_x * rand_x 

            # k_t_minus_1 = c0 * (k_t_minus_05 - c1 * pred_l) + sigmas * rand_k
            k_t_minus_1 = c1 *k_t_minus_05 + c0 * pred_l + sigmas * rand_k
            k_t_minus_1 = self.crystal_family.proj_k_to_spacegroup(k_t_minus_1, sg_t_minus_1)
            l_t_minus_1 = self.crystal_family.v2m(k_t_minus_1)

            t_t_minus_05, _ = to_dense_batch(t_t_minus_05, batch, fill_value=0)
            symm_t_minus_05, _ = to_dense_batch(symm_t_minus_05, batch, fill_value=0)
            t_t_minus_1, symm_t_minus_1 = self.discrete_noise.sample_zs_from_zt_and_pred(t_t_minus_05, symm_t_minus_05, pred_t, pred_symm, times, times-1, node_mask, sg_t, sg_t_minus_1)
            t_t_minus_1 = t_t_minus_1[node_mask]
            symm_t_minus_1 = symm_t_minus_1[node_mask]

            traj[t - 1] = {
                'num_atoms' : num_atoms,
                'atom_types' : t_t_minus_1,
                'site_symm' : symm_t_minus_1,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1,
                'ks' : k_t_minus_1,
                'spacegroup': sg_t_minus_1,
            }
        # traj_stack = {
        #     'num_atoms' : num_atoms,
        #     'atom_types' : torch.stack([traj[i]['atom_types'] for i in range(self.beta_scheduler.timesteps, -1, -1)]).argmax(dim=-1) + 1,
        #     'site_symm' : torch.stack([traj[i]['site_symm'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        #     'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        #     'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        #     'all_ks': torch.stack([traj[i]['ks'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        #     'all_spacegroup': torch.stack([traj[i]['spacegroup'] for i in range(self.beta_scheduler.timesteps, -1, -1)]),
        # }
        traj_stack = None
        # print(f'theory_steps: {theory_steps}')
        # drop all dummy elements (atom types = MAX_ATOMIC_NUM)
        dummy_ind = (traj[0]['atom_types'].argmax(dim=-1) == self.discrete_noise.max_atomic_num).long()
        traj[0]['frac_coords'] = traj[0]['frac_coords'][(1 - dummy_ind).bool()]
        traj[0]['atom_types'] = traj[0]['atom_types'][(1 - dummy_ind).bool()]
        traj[0]['site_symm'] = traj[0]['site_symm'][(1 - dummy_ind).bool()]
        # find for each crystal how many non-dummy atoms are there
        traj[0]['num_atoms'] = find_num_atoms(dummy_ind, num_atoms).to(self.device)
        # remove lattices and ks for empty crystals corresponding to num_atoms = 0
        empty_crystals = (traj[0]['num_atoms'] == 0).long()
        traj[0]['ks'] = traj[0]['ks'][(1 - empty_crystals).bool()]
        traj[0]['lattices'] = traj[0]['lattices'][(1 - empty_crystals).bool()]
        print(f"Number of empty crystals generated: {empty_crystals.sum().item()}/{batch_size}")
        # use predicted site symmetry to create copies of atoms
        # frac coords, atom types and num atoms removed for empty crystals in modify_frac_coords()
        # note, depend on symmetry group,the number of atoms may multipled a lot
        traj[0] = modify_frac_coords(traj[0], traj[0]['spacegroup'], traj[0]['num_atoms'])
        print("INFO: Done modifying fractional coordinates.")
        return traj[0], traj_stack


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)
        loss_rate = output_dict['loss_rate']
        loss_sg = output_dict['loss_sg']
        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss_symm = output_dict['loss_symm']
        loss = output_dict['loss']


        self.log_dict(
            {'train_loss': loss,
            'lattice_loss': loss_lattice,
            'coord_loss': loss_coord,
            'type_loss': loss_type,
            'symm_loss': loss_symm,
            'sg_loss': loss_sg,
            'rate_loss': loss_rate,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if loss.isnan():
            return None

        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # self.simple_gen_evaluation()
        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='val')

        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        if (self.current_epoch + 1) % self.hparams.data.eval_every_epoch == 0 and batch_idx == 0:
            # run a simpler evaluation
            self.simple_gen_evaluation()

        return loss

    def simple_gen_evaluation(self):
        
        frac_coords = []
        num_atoms = []
        atom_types = []
        lattices = []
        site_symmetries = []
        spacegroups = []

        outputs, traj = self.sample(self.hparams.data.eval_generate_samples, step_lr = 1e-5)
        del traj
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())
        site_symmetries.append(outputs['site_symm'].detach().cpu())
        spacegroups.append(outputs['spacegroup'].detach().cpu())
        del outputs

        frac_coords = torch.cat(frac_coords, dim=0)
        num_atoms = torch.cat(num_atoms, dim=0)
        atom_types = torch.cat(atom_types, dim=0)
        lattices = torch.cat(lattices, dim=0) 
        site_symmetries = torch.cat(site_symmetries, dim=0)
        spacegroups = torch.cat(spacegroups, dim=0)
        lengths, angles = lattices_to_params_shape(lattices)
        
        # generated crystals
        # kwargs = {"site_symmetries": site_symmetries, "spacegroups": spacegroups}
        pred_crys_array_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
        gen_crys = p_map(lambda x: Crystal(x), pred_crys_array_list, num_cpus=4)
        print(f"INFO: Done generating {self.hparams.data.eval_generate_samples} crystals (Epoch: {self.current_epoch + 1})")
        
        # ground truth crystals
        if os.path.exists(self.hparams.data.datamodule.datasets.val[0].gt_crys_path):
            gt_crys = torch.load(self.hparams.data.datamodule.datasets.val[0].gt_crys_path)
        else:
            csv = pd.read_csv(self.hparams.data.datamodule.datasets.val[0].path)
            gt_crys = t_map(get_gt_crys_ori, csv['cif'])
            torch.save(gt_crys, self.hparams.data.datamodule.datasets.val[0].gt_crys_path)
            
        print(f"INFO: Done reading ground truth crystals (Epoch: {self.current_epoch + 1})")
        
        gen_evaluator = GenEval(gen_crys, gt_crys, n_samples=0, eval_model_name=self.hparams.data.eval_model_name,
                                gt_prop_eval_path=self.hparams.data.datamodule.datasets.val[0].gt_prop_eval_path)
        gen_metrics = gen_evaluator.get_metrics()
        print(gen_metrics)
        print(f"INFO: Done generation test (Epoch: {self.current_epoch + 1})")
        self.log_dict(gen_metrics)

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        output_dict = self(batch)

        log_dict, loss = self.compute_stats(output_dict, prefix='test')

        self.log_dict(
            log_dict,
        )
        return loss

    def compute_stats(self, output_dict, prefix):
        loss_rate = output_dict['loss_rate']
        loss_sg = output_dict['loss_sg']
        loss_lattice = output_dict['loss_lattice']
        loss_coord = output_dict['loss_coord']
        loss_type = output_dict['loss_type']
        loss_symm = output_dict['loss_symm']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_lattice_loss': loss_lattice,
            f'{prefix}_coord_loss': loss_coord,
            f'{prefix}_type_loss': loss_type,
            f'{prefix}_symm_loss': loss_symm,
            f'{prefix}_sg_loss': loss_sg,
            f'{prefix}_rate_loss': loss_rate,
        }

        return log_dict, loss