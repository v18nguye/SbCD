from collections import Counter
import argparse
import os
import json
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import pandas as pd
import torch
import pdb

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import sys
sys.path.append('.')
sys.path.append('..')
import os
myDir = os.getcwd()
sys.path.append(myDir)

from sbcd.common.data_utils import build_crystal, get_symmetry_info
from scripts.crystal import Crystal
from pyxtal import pyxtal

warnings.simplefilter("ignore")
from scripts.eval_utils import (
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov)

COV_Cutoffs = {
    'mp20': {'struc': 0.4, 'comp': 10.},
    'carbon': {'struc': 0.2, 'comp': 4.},
    'perovskite': {'struc': 0.2, 'comp': 4},
}

NUM_CPUS = 4

class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None
        validity = [c1.valid and c2.valid for c1,c2 in zip(self.preds, self.gts)]

        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(
                self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}     

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class RecEvalBatch(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(
                    pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = process_one(self.preds[j][i], self.gts[i], self.preds[j][i].valid)
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))
            
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {'match_rate': match_rate,
                'rms_dist': mean_rms_dist}    

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics
    
def plot_sg_hist(gen_crys, test_crys, method='SbCD', dataname='MP20 Test'):
    '''plot sg hist of gen and test crystals
    '''
    import matplotlib.pyplot as plt
    import numpy as np

    pred_spacegroup = [c.real_spacegroup for c in gen_crys]
    gt_spacegroup = [c.spacegroup for c in test_crys]
    print(f'pred_spacegroup: {len(pred_spacegroup)}, gt_spacegroup: {len(gt_spacegroup)}')
    # Convert list of spacegroups into distribution over 230 possible spacegroups
    gt_hist = np.histogram(gt_spacegroup, bins=range(1, 231))[0]/len(gt_spacegroup)
    pred_hist = np.histogram(pred_spacegroup, bins=range(1, 231))[0]/len(pred_spacegroup)

    # Bin centers for 230 space groups
    bin_centers = np.arange(1, 230)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafafa')

    colors = ['#3D405B', '#81B29A']

    # Filled step histograms
    ax.fill_between(bin_centers, gt_hist, step='mid', alpha=0.4,
                    color=colors[0], label=dataname, zorder=2)
    ax.fill_between(bin_centers, pred_hist, step='mid', alpha=0.4,
                    color=colors[1], label=method, zorder=2)

    # Outline steps
    ax.step(bin_centers, gt_hist, where='mid', color=colors[0], linewidth=2, zorder=3)
    ax.step(bin_centers, pred_hist, where='mid', color=colors[1], linewidth=2, zorder=3)

    # Labels
    ax.set_xlabel('Space Group Number', fontweight='bold', fontsize=20,
                labelpad=12, color='#333333')
    ax.set_ylabel('Normalized Frequency', fontweight='bold', fontsize=20,
                labelpad=12, color='#333333')
    ax.set_xlim(0, 231)
    ax.tick_params(axis='both', labelsize=16, colors='#333333')

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')

    ax.grid(axis='y', linestyle='--', alpha=0.2, color='black')
    ax.set_axisbelow(True)

    # Legend below
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2,
                    frameon=True, fancybox=True, edgecolor='#cccccc',
                    facecolor='white', framealpha=0.95, fontsize=16)
    legend.get_frame().set_linewidth(1.5)

    fig.tight_layout()
    fig.savefig('sg_mp20.pdf', dpi=300, bbox_inches='tight', format='pdf')
    fig.savefig('sg_mp20.png', dpi=300, bbox_inches='tight')

class GenEval(object):

    def __init__(self, pred_crys, gt_crys, n_samples=1000, eval_model_name=None, gt_prop_eval_path=None):
        self.crys = pred_crys
        self.gt_crys = gt_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name
        self.gt_prop_eval_path = gt_prop_eval_path

        valid_crys = [c for c in pred_crys if c.valid]
        # # plot sg hists & ehull
        # plot_sg_hist(valid_crys, gt_crys)
        # assert 1 == 2
        if n_samples == 0:
            # use all valid crystals instead
            self.valid_samples = valid_crys
        elif len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f'not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}')

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.crys]).mean()
        valid = np.array([c.valid for c in self.crys]).mean()
        return {'comp_valid': comp_valid,
                'struct_valid': struct_valid,
                'valid': valid}


    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        print(f"mean pred_densities: {np.median(pred_densities)}")
        gt_densities = [c.structure.density for c in self.gt_crys]
        print(f"mean gt_densities: {np.median(gt_densities)}")
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {'wdist_density': wdist_density}


    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.gt_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {'wdist_num_elems': wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                                         c.dict for c in self.valid_samples])
            if self.gt_prop_eval_path is not None and os.path.exists(self.gt_prop_eval_path):
                gt_props = torch.load(self.gt_prop_eval_path)
            else:
                gt_props = prop_model_eval(self.eval_model_name, [
                                           c.dict for c in self.gt_crys])
                torch.save(gt_props, self.gt_prop_eval_path)
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {'wdist_prop': wdist_prop}
        else:
            return {'wdist_prop': None}

    def get_spacegroup_wdist(self):
        pred_spacegroup = [c.real_spacegroup for c in self.valid_samples]
        gt_spacegroup = [c.spacegroup for c in self.gt_crys]
        wdist_spacegroup = wasserstein_distance(pred_spacegroup, gt_spacegroup)
        return {'wdist_spacegroup': wdist_spacegroup}

    def get_spacegroup_js(self):
        pred_spacegroup = [c.real_spacegroup for c in self.valid_samples]
        gt_spacegroup = [c.spacegroup for c in self.gt_crys]
        # Convert list of spacegroups into distribution over 230 possible spacegroups

        gt_hist = np.histogram(gt_spacegroup, bins=range(1, 231))[0]/len(gt_spacegroup)
        pred_hist = np.histogram(pred_spacegroup, bins=range(1, 231))[0]/len(pred_spacegroup)
        return {"js_spacegroup": jensenshannon(gt_hist, pred_hist)}


    def get_spacegroup_match(self):
        spacegroup_match = np.array([c.spacegroup_match for c in self.valid_samples]).mean()
        return {'spacegroup_match': spacegroup_match}


    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.crys, self.gt_crys,
            struc_cutoff=cutoff_dict['struc'],
            comp_cutoff=cutoff_dict['comp'])
        return cov_metrics_dict

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        if len(self.valid_samples) == 0:
            print("No valid crystals generated")
            metrics['wdist'] = 5
            return metrics
        metrics.update(self.get_density_wdist())
        # metrics.update(self.get_prop_wdist())
        metrics.update(self.get_num_elem_wdist())
        metrics['wdist'] = metrics['wdist_num_elems']*metrics['wdist_density']
        metrics.update(self.get_coverage())
        metrics.update(self.get_spacegroup_wdist())
        try:
            metrics.update(self.get_spacegroup_js())
        except:
            print("Spacegroup JS didn't work")
        metrics.update(self.get_spacegroup_match())
        return metrics


def get_file_paths(root_path, task, label='', suffix='pt'):
    if label == '':
        out_name = f'eval_{task}.{suffix}'
    else:
        out_name = f'eval_{task}_{label}.{suffix}'
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0, get_true_list=True):
    data = load_data(file_path)
    if batch_idx == -1:
        batch_size = data['frac_coords'].shape[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data['frac_coords'][i],
                data['atom_types'][i],
                data['lengths'][i],
                data['angles'][i],
                data['num_atoms'][i])
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        kwargs = dict()
        if 'spacegroups' in data and 'site_symmetries' in data:
            kwargs['spacegroups'] = data['spacegroups']
            kwargs['site_symmetries'] = data['site_symmetries']
        crys_array_list = get_crystals_list(
            data['frac_coords'],
            data['atom_types'],
            data['lengths'],
            data['angles'],
            data['num_atoms'],
            **kwargs)        
    else:
        crys_array_list = get_crystals_list(
            data['frac_coords'][batch_idx],
            data['atom_types'][batch_idx],
            data['lengths'][batch_idx],
            data['angles'][batch_idx],
            data['num_atoms'][batch_idx])

    if get_true_list and 'input_data_batch' in data:
        batch = data['input_data_batch']
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch['frac_coords'], batch['atom_types'], batch['lengths'],
                batch['angles'], batch['num_atoms'])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif,fmt='cif')
    spacegroup = structure.get_space_group_info()[1]
    lattice = structure.lattice
    crys_array_dict = {
        'frac_coords':structure.frac_coords,
        'atom_types':np.array([_.Z for _ in structure.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles),
        'spacegroups': spacegroup
    }
    return Crystal(crys_array_dict)

def get_gt_crys_ori_conventional(cif):
    crystal = build_crystal(cif)
    crystal, sym_info, dummy_repr_ind, dummy_origin_ind, identifier = get_symmetry_info(crystal, tol=0.01, num_repr=0)
    spacegroup = sym_info['spacegroup']
    lattice = crystal.lattice
    crys_array_dict = {
        'frac_coords':crystal.frac_coords,
        'atom_types':np.array([_.Z for _ in crystal.species]),
        'lengths': np.array(lattice.abc),
        'angles': np.array(lattice.angles),
        'spacegroups': spacegroup
    }
    return Crystal(crys_array_dict) 

# class CrystalWPTemplates():
#     '''
#     Class for checking Wyckoff Templates of crystals
#     '''
#     def __init__(self, gt_crys, gt_crys_file):
#         '''
#         Gets all unique wyckoff templates for ground truth crystals
#         '''
#         if 'train' not in gt_crys_file:
#             raise ValueError('Must use the train dataset!')
#         gt_wp_file = gt_crys_file.split('train_gt_crys.pt')[0]+'train_gt_crys_wp.pt'

#         if os.path.exists(gt_wp_file):
#             print("Loading gt_crys_wp")
#             wp_frozen_labels_gt = torch.load(gt_wp_file)    
#         else:
#             gt_crys_dict = []
#             for crys in tqdm(gt_crys):
#                 crys_dict = {
#                     'lengths': crys.lengths,
#                     'angles': crys.angles,
#                     'atom_types': crys.atom_types,
#                     'frac_coords': crys.frac_coords,
#                 }
#                 gt_crys_dict.append(crys_dict)
#             wp_labels_gt = []
#             for crys in tqdm(gt_crys_dict):
#                 if crys is None:
#                     continue
#                 try:
#                     wp_labels = self.get_wp_label_counter(crys)
#                 except:
#                     continue
#                 if wp_labels is None:
#                     continue
#                 wp_labels_gt.append(wp_labels)
#             wp_frozen_labels_gt = [self.hashable_wp_labels(*wp_labels) for wp_labels in wp_labels_gt]
#             try:
#                 torch.save(wp_frozen_labels_gt, gt_wp_file)
#             except Exception as e:
#                 print(e)
#                 print("gt_crys_wp couldn't be saved")
#         self.wp_frozen_labels_gt_counter = Counter(wp_frozen_labels_gt)

#     def crys_dict_to_pymatgen(self, crys_dict):
#         lengths = crys_dict['lengths']
#         angles = crys_dict['angles']
#         lattice = Lattice.from_parameters(*lengths, *angles)
#         if len(crys_dict['atom_types'].shape) == 2:
#             species = crys_dict['atom_types'].argmax(-1) + 1
#         else:
#             species = crys_dict['atom_types']
#         coords = crys_dict['frac_coords']
#         try:
#             structure = Structure(lattice, species, coords)
#             return structure
#         except:
#             return None

#     def get_wp_label_counter(self, crys):
#         structure = self.crys_dict_to_pymatgen(crys)
#         if structure is None:
#             return None
#         if np.isnan(structure.lattice.angles).any() or np.isnan(structure.lattice.lengths).any():
#             return None
#         if np.isinf(structure.lattice.angles).any() or np.isinf(structure.lattice.lengths).any():
#             return None
#         try:
#             spga = SpacegroupAnalyzer(structure, symprec=0.1)
#         except:
#             return None
#         crystal = spga.get_refined_structure()

#         pyx = pyxtal()
#         try:
#             pyx.from_seed(crystal)
#             wp_labels = []
#             for atom_site in pyx.atom_sites:
#                 wp_labels.append(atom_site.wp.get_label())
            
#             return pyx.group.number, Counter(wp_labels)
#         except:
#             return None

#     def hashable_wp_labels(self, symm_group, wp_labels):
#         return (symm_group, frozenset(wp_labels.items()))

#     def count_labels_in_gt(self, crys_array_list):
#         '''
#         Returns the number of unique wyckoff templates in the list of crystals, the number 
#         of these templates that appear in the ground truth list, and the proportion in the ground truth.
#         '''
#         wp_labels = []
#         for crys in tqdm(crys_array_list):
#             try:
#                 wp_labels.append(self.get_wp_label_counter(crys))
#             except Exception as e:
#                 print(e)
#                 print('can not get wp labels')
#         wp_frozen_labels = [self.hashable_wp_labels(*wp_labels_list) for wp_labels_list in wp_labels  if wp_labels_list is not None]
#         wp_frozen_labels_counter = Counter(wp_frozen_labels)

#         label_in_gt = []
#         for wp_labels in tqdm(wp_frozen_labels_counter.keys()):
#             label_in_gt.append(wp_labels in self.wp_frozen_labels_gt_counter)
#         label_in_gt = np.array(label_in_gt)

#         return len(wp_frozen_labels_counter), label_in_gt.sum(), label_in_gt.mean() # unique labels, wp labels in gts, % in gts
    
def get_proxy_metric(args):
    '''get standard proxy metric for de novo crystal generation.
    '''
    all_metrics = {}

    cfg = load_config(args.root_path)
    eval_model_name = cfg.data.eval_model_name

    if 'gen' in args.tasks:

        gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
        crys_array_list, _ = get_crystal_array_list(gen_file_path, batch_idx = args.batch_idx, get_true_list=False)
        print(f'Evaluate {len(crys_array_list)} samples.')
        if args.gt_crys_file != '':
            if os.path.exists(args.gt_crys_file):
                print(f"Loading gt_crys: {args.gt_crys_file}")
                gt_crys = torch.load(args.gt_crys_file)    
            else:
                print(f"Reading gt_crys csv: {args.gt_file}")
                csv = pd.read_csv(args.gt_file)
                gt_crys = p_map(get_gt_crys_ori_conventional, csv['cif'])
                try:
                    torch.save(gt_crys, args.gt_crys_file)
                except Exception as e:
                    print(e)
                    print("gt_crys couldn't be saved")
        elif args.gt_file != '':
            print("Reading gt_crys csv")
            csv = pd.read_csv(args.gt_file)
            if args.conventional: # TODO: check this parameter
                gt_crys = p_map(get_gt_crys_ori_conventional, csv['cif'])
            else:
                gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            # always ground gt_file is provided
            # if not provided then only use the reconstruction path to load true crystals
            recon_file_path = get_file_paths(args.root_path, 'recon', args.label)
            _, true_crystal_array_list = get_crystal_array_list(
                recon_file_path)
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

        if os.path.exists(args.root_path + f'/gen_crys_{args.label}.pt'):
            print(f"Loading gen_crys_{args.label}.pt")
            gen_crys = torch.load(args.root_path + f'/gen_crys_{args.label}.pt')
        elif args.label == '' and os.path.exists(args.root_path + '/gen_crys.pt'):
            gen_crys = torch.load(args.root_path + '/gen_crys.pt')
        else:
            # Mapping gen_crys
            gen_crys = p_map(lambda x: Crystal(x), crys_array_list, num_cpus=NUM_CPUS)
            try:
                torch.save(gen_crys, args.root_path + f'/gen_crys_{args.label}.pt')
            except Exception as e:
                print(e)
                print("Gen_crys couldn't be saved")
        gt_prop_eval_path = args.gt_prop_eval_path if args.gt_prop_eval_path is not None else cfg.data.datamodule.datasets.test[0].gt_prop_eval_path
        print(f"loading gt_prop_eval_path: {gt_prop_eval_path}")
        print(f"args.n_samples: {args.n_samples}")
        gen_evaluator = GenEval(
            gen_crys, gt_crys, eval_model_name=eval_model_name, n_samples= args.n_samples,
            gt_prop_eval_path=gt_prop_eval_path)
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    else:

        recon_file_path = get_file_paths(args.root_path, 'diff', args.label)
        batch_idx = -1 if args.multi_eval else 0
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path, batch_idx = batch_idx)
        if args.gt_file != '':
            csv = pd.read_csv(args.gt_file)
            gt_crys = p_map(get_gt_crys_ori, csv['cif'])
        else:
            gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

        if not args.multi_eval:
            pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
        else:
            pred_crys = []
            for i in range(len(crys_array_list)):
                print(f"Processing batch {i}")
                pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i]))

        if 'csp' in args.tasks: 

            if args.multi_eval:
                rec_evaluator = RecEvalBatch(pred_crys, gt_crys)
            else:
                rec_evaluator = RecEval(pred_crys, gt_crys)

            recon_metrics = rec_evaluator.get_metrics()

            all_metrics.update(recon_metrics)

    print(all_metrics)

    if args.label == '':
        metrics_out_file = 'eval_proxy_metrics.json'
    else:
        metrics_out_file = f'eval_proxy_metrics_{args.label}.json'
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, 'r') as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, 'w') as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, 'w') as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, 'w') as f:
            json.dump(all_metrics, f)

# def get_wynckoff_template_metric(args):
#     gen_file_path = get_file_paths(args.root_path, 'gen', args.label)
#     crys_array_list, _ = get_crystal_array_list(gen_file_path, batch_idx = args.batch_idx, get_true_list=False)

#     if args.gt_crys_file != '':
#         if os.path.exists(args.gt_crys_file):
#             print("Loading gt_crys")
#             gt_crys = torch.load(args.gt_crys_file)    
#         else:
#             print("Reading gt_crys csv")
#             csv = pd.read_csv(args.gt_file)
#             gt_crys = p_map(get_gt_crys_ori_conventional, csv['cif'])
#             try:
#                 torch.save(gt_crys, args.gt_crys_file)
#             except Exception as e:
#                 print(e)
#                 print("gt_crys couldn't be saved")
#     else:
#         raise ValueError('gt_crys_file should not be None')
#     print(f'Evaluate {len(crys_array_list)} samples.')
#     wp_temp_evaluator = CrystalWPTemplates(gt_crys, args.gt_crys_file)
#     res = wp_temp_evaluator.count_labels_in_gt(crys_array_list)
#     print(res)    
    
def main(args):
    get_proxy_metric(args)
    # get_wynckoff_template_metric(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', required=True)
    parser.add_argument('--label', default='')
    parser.add_argument('--tasks', nargs='+', default=['csp', 'gen'])
    parser.add_argument('--gt_file',default='')
    parser.add_argument('--gt_crys_file',default='')
    parser.add_argument('--multi_eval',action='store_true')
    parser.add_argument('--n_samples', type=int, default=1000)
    parser.add_argument('--conventional', type=bool, default=False,
                        help='whether to use the conventional lattice instead of the primitive lattice')
    parser.add_argument('--batch_idx', type=int, default=-2,
                        help='default -2, should be -1 for flowmm')
    parser.add_argument('--gt_prop_eval_path', default=None)
    args = parser.parse_args()
    main(args)