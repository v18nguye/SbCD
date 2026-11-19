import torch
import torch.nn as nn
import numpy as np
import math

from sbcd.pl_modules.lattice.matrix import logm, expm, sqrtm

class CrystalFamily(nn.Module):
    '''converting between k and L constrained by crystal systems.
    '''

    def __init__(self):

        super(CrystalFamily, self).__init__()

        basis = self.get_basis()
        masks, biass = self.get_spacegroup_constraints()
        family = self.get_family_idx()

        self.register_buffer('basis', basis)
        self.register_buffer('masks', masks)
        self.register_buffer('biass', biass)
        self.register_buffer('family', family)

    def get_basis(self):

        basis = torch.FloatTensor([
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ])

        # Normalize
        basis = basis / basis.norm(dim=(-1,-2)).unsqueeze(-1).unsqueeze(-1)

        return basis

    def get_spacegroup_constraint(self, spacegroup):
        '''k constraint
        '''
        mask = torch.ones(6)
        bias = torch.zeros(6)

        if 195 <= spacegroup <= 230:
            pos = [0,1,2,3,4]
            mask[pos] = 0.

        elif 143 <= spacegroup <= 194:
            pos = [0,1,2,3]
            mask[pos] = 0.
            bias[0] = -0.25 * np.log(3) * np.sqrt(2)

        elif 75 <= spacegroup <= 142:
            pos = [0,1,2,3]
            mask[pos] = 0.

        elif 16 <= spacegroup <= 74:
            pos = [0,1,2]
            mask[pos] = 0.

        elif 3 <= spacegroup <= 15:
            pos = [0,2]
            mask[pos] = 0.

        elif 0 <= spacegroup <= 2:
            pass

        return mask, bias

    def get_spacegroup_constraints(self):

        masks, biass = [], []

        for i in range(231):
            mask, bias = self.get_spacegroup_constraint(i)
            masks.append(mask.unsqueeze(0))
            biass.append(bias.unsqueeze(0))

        return torch.cat(masks, dim = 0), torch.cat(biass, dim = 0)

    def get_family_idx(self):
        '''Mapping space group to crystal systems.
        '''

        family = []
        for spacegroup in range(231):
            if 195 <= spacegroup <= 230:
                family.append(7)
            
            elif 168 <= spacegroup <= 194:
                family.append(6)

            elif 143 <= spacegroup <= 167:
                family.append(5)

            elif 75 <= spacegroup <= 142:
                family.append(4)

            elif 16 <= spacegroup <= 74:
                family.append(3)

            elif 3 <= spacegroup <= 15:
                family.append(2)

            elif 0 <= spacegroup <= 2:
                family.append(1)

        return torch.LongTensor(family)

    def de_so3(self, L):
        '''map L to symmetric postive definite (P) of 
        its polar decomposition, which is L=QP, WHERE P=exp(S), ref: chatgpt

        So  P only captures the symmetric positive definite (stretching) part of L, and discards the orthogonal (rotation/reflection) part.
        '''

        # L: B * 3 * 3

        LLT = L @ L.transpose(-1,-2)
        L_sym = sqrtm(LLT) # this is P, in L=QP
        return L_sym # P

    def v2m(self, vec): # vector to matrix
        '''map k to P=exp(S), with S=log_mat
        '''

        batch_size, dims = vec.shape
        if dims == 6:
            basis = self.basis
        elif dims == 5:
            basis = self.basis[:-1]
        log_mat = torch.einsum('bk, kij -> bij', vec, basis) # proposition 2
        mat = expm(log_mat) # proposition 1, return L
        return mat

    def m2v(self, mat): # matrix to vector
        '''map P=exp(S), in L=QP, to k
        '''

        # mat: B * 3 * 3

        log_mat = logm(mat) # return S
        vec = torch.einsum('bij, kij -> bk', log_mat, self.basis) # from S, B, extract k
        return vec # k

    def proj_k_to_spacegroup(self, vec, spacegroup):
        batch_size, dims = vec.shape
        if dims == 6:
            masks = self.masks[spacegroup, :] # B * 6
            biass = self.biass[spacegroup, :] # B * 6  
        elif dims == 5:
            # - volume
            masks = self.masks[spacegroup, :-1] # B * 5
            biass = self.biass[spacegroup, :-1] # B * 5            
        return vec * masks + biass # space group constraint, masking out some ks to satisfy a specific crystal system, return vec

    def get_sg_bias(self, sg):
        return self.biass[sg, :]

    def get_sg_mask(self, sg):
        return self.masks[sg, :]