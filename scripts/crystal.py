from collections import Counter
import warnings
import numpy as np
import torch
import pdb

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.structure import SiteStatsFingerprint
from matminer.featurizers.composition.composite import ElementProperty

warnings.simplefilter("ignore")
from scripts.eval_utils import (
    smact_validity, structure_validity)

Crystal_Tol = 0.1
CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CrystalNNFP_full = SiteStatsFingerprint(
    CrystalNNFingerprint.from_preset('ops'),
    stats=('mean', 'maximum', 'minimum', 'std_dev'))
CrystalNNFP_plus = SiteStatsFingerprint(
    CrystalNNFingerprint.from_preset('ops'),
    stats=('mean', 'maximum'))

CompFP = ElementProperty.from_preset('magpie')


# Tools from FlowMM
def bits2int(x, out_dtype):
    """Converts bits x in (..., n) into an integer in (...)."""
    if isinstance(x, torch.Tensor):
        x = x.to(dtype=out_dtype)
        n = x.size(-1)
        x = torch.sum(x * (2 ** torch.arange(n)), dim=-1)
        return x
    else:
        x = x.astype(out_dtype)
        n = x.shape[-1]
        x = np.sum(x * (2 ** np.arange(n)), axis=-1)
        return x

def analog_bits_to_int(analog_bits: torch.Tensor ) -> torch.LongTensor:
    if isinstance(analog_bits, torch.Tensor):
        signs = analog_bits.sign()
        bits = (signs + 1) / 2
        return bits2int(bits, torch.long)
    else:
        signs = np.sign(analog_bits)
        bits = (signs + 1) / 2
        return bits2int(bits, np.int64)


class Crystal(object):

    def __init__(self, crys_array_dict, full_fingerprint=False):
        self.frac_coords = crys_array_dict['frac_coords']
        self.atom_types = crys_array_dict['atom_types']
        self.lengths = crys_array_dict['lengths']
        self.angles = crys_array_dict['angles']
        self.full_fingerprint = full_fingerprint
        if 'spacegroups' in crys_array_dict:
            self.spacegroup = crys_array_dict['spacegroups']
        else:
            self.spacegroup = None
        self.invalid_reason = None
        
        # check for NaN values 
        if np.isnan(self.lengths).any() or np.isinf(self.lengths).any():
            self.lengths = np.array([1, 1, 1]) * 100
            crys_array_dict['lengths'] = self.lengths
            
            self.constructed = False
            self.invalid_reason = 'nan_value_0'
        
        self.dict = crys_array_dict
        if len(self.atom_types.shape) > 1:
            if self.atom_types.shape[-1] == 7:
                # Analog bits, from flowmm
                atom_types = analog_bits_to_int(self.atom_types) + 1
            else:
                # this implies the distribution over atom_types is passed instead of the atom_types
                atom_types =  np.argmax(self.atom_types, axis=-1) + 1
            self.dict['atom_types'] = atom_types
            self.atom_types = atom_types
    
        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()
        self.get_symmetry()


    def get_structure(self):
        # if (1 > self.atom_types).any() or (self.atom_types > 94).any():
        #     self.constructed = False
        #     self.invalid_reason = f"{self.atom_types=} are not with range"
        # if len(self.frac_coords) > 30:
        #     self.constructed = False
        #     self.invalid_reason = 'too_many_atoms'
        if len(self.atom_types) == 0:
            self.constructed = False
            self.invalid_reason = 'empty'
        elif min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = 'non_positive_lattice'
        elif np.isnan(self.lengths).any() or np.isnan(self.angles).any() or  np.isnan(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'nan_value_1'
        elif np.isinf(self.lengths).any() or np.isinf(self.angles).any() or  np.isinf(self.frac_coords).any():
            self.constructed = False
            self.invalid_reason = 'inf_value_2'
        # elif (self.lengths > 1000).any():
        #     self.constructed = False
        #     self.invalid_reason = 'bad_value_3'
        # elif (self.angles >= 180).any() or (self.angles <= 0).any():
        #     self.constructed = False
        #     self.invalid_reason = 'bad_value_4'
        # elif (self.frac_coords > 1).any() or (self.frac_coords < 0).any():
        #     self.constructed = False
        #     self.invalid_reason = 'bad_value_5'
        # elif self.lengths.min() < 1:
        #     self.constructed = False
        #     self.invalid_reason = 'bad_value_6'
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
                # if self.structure.volume < 0.1:
                #     self.constructed = False
                #     self.invalid_reason = 'unrealistically_small_lattice'

            except Exception as e:
                self.constructed = False
                self.invalid_reason = 'construction_raises_exception'
                print(f'exception: {e}')
        if not self.constructed:
            print(f'self.invalid_reason: {self.invalid_reason}')

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        if len(elem_counter) == 0:
            self.elems = ()
            self.comps = ()
            return
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype('int').tolist())

    def get_validity(self):
        if self.constructed:
            if len(self.elems) == 0:
                self.comp_valid = False
            else:
                self.comp_valid = smact_validity(self.elems, self.comps)
            self.struct_valid = structure_validity(self.structure)
        else:
            self.comp_valid = False
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        if len(self.atom_types) == 0:
            self.struct_fp = None
            self.comp_fp = None
            return
        if not self.constructed:
            self.struct_fp = None
            self.comp_fp = None
            return 
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        if self.full_fingerprint:
            try:
                struct_fp = CrystalNNFP_full.featurize(self.structure)
                self.struct_fp = np.array(struct_fp)
            except Exception:
                self.valid = False
                self.comp_fp = None
                self.struct_fp = None
        else:
            try:
                site_fps = [CrystalNNFP.featurize(
                    self.structure, i) for i in range(len(self.structure))]
            except Exception:
                # counts crystal as invalid if fingerprint cannot be constructed.
                self.valid = False
                self.comp_fp = None
                self.struct_fp = None
                return
            self.struct_fp = np.array(site_fps).mean(axis=0)

    def get_symmetry(self):
        if self.constructed:
            try:
                spga = SpacegroupAnalyzer(self.structure, symprec=Crystal_Tol)
                self.real_spacegroup = spga.get_space_group_number()
            except Exception:
                self.real_spacegroup = 1
            if self.real_spacegroup is None:
                # Default to setting to 1
                self.real_spacegroup = 1
        else:
            self.real_spacegroup = None
        self.spacegroup_match = self.real_spacegroup == self.spacegroup