"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import io, os
from argparse import ArgumentParser, Namespace
from contextlib import redirect_stdout
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import json
import torch
from matplotlib.patches import Rectangle
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from toolz import compose

# import py3Dmol
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt
from ase.data import colors

import sys
sys.path.append('.')

from scripts.dft_create_inputs import filter_prerelaxed, maybe_get_missing_columns, get_intersection
from scripts.dft_create_inputs import COLUMNS_COMPUTATIONS, to_structure, get_chemsys
from scripts.joblib_ import joblib_map

from scripts.tabular import VALID_STAGES, VALID_TABULAR_DATASETS, get_tabular_dataset
import pdb

trap = io.StringIO()

def plot_stable_struc(struct):
    save_path = '/home/cubi/research/codes/sbcd/plot_figs/'
    spg = SpacegroupAnalyzer(struct)
    symbol = spg.get_space_group_symbol()
    number = spg.get_space_group_number()
    reduced_formula = struct.composition.reduced_formula

    # convert to ASE atoms
    atoms = AseAtomsAdaptor.get_atoms(struct)

    # unique element -> color
    element_colors = {}
    for atom in atoms:
        if atom.symbol not in element_colors:
            element_colors[atom.symbol] = colors.jmol_colors[atom.number]

    # plot
    fig, ax = plt.subplots(figsize=(5,5))
    plot_atoms(atoms, ax, rotation='20x,30y,0z')

    ax.set_axis_off()
    # plot_atoms(atoms, ax, radii=0.4)
    elements = list(element_colors.items())
    n = len(elements)
    # leave space below
    fig.subplots_adjust(bottom=0.2)

    # box settings
    box_w = 0.06
    box_h = 0.06
    gap = 0.02

    # total width → center horizontally
    total_w = n * box_w + (n - 1) * gap
    start_x = 0.5 - total_w / 2
    y = 0.07

    for i, (elem, rgb) in enumerate(elements):
        x = start_x + i * (box_w + gap)

        # draw box
        rect = Rectangle(
            (x, y), box_w, box_h,
            transform=fig.transFigure,
            facecolor=rgb,
            edgecolor='black',
            linewidth=1.0
        )
        fig.add_artist(rect)

        # choose text color (white on dark, black on light)
        brightness = sum(rgb) / 3
        text_color = 'black' #if brightness > 0.5 else 'white'

        # element label INSIDE box
        fig.text(
            x + box_w / 2,
            y + box_h / 2,
            elem,
            ha='center',
            va='center',
            fontsize=12,
            color=text_color,
            weight='bold'
        )

    name_file = f'{number}_{reduced_formula}'
    plt.savefig(f"{save_path}{name_file}.png")

    # save structure
    with open(f"{save_path}{name_file}.json", "w") as f:
        json.dump(struct.as_dict(), f, indent=2)

def save_metrics_only_overwrite_newly_computed(
    path: Path, metrics: dict[str, float]
) -> None:
    # only overwrite metrics computed in the new run.
    if Path(path).exists():
        with open(path, "r") as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(metrics)
            else:
                with open(path, "w") as f:
                    json.dump(metrics, f)
        if isinstance(written_metrics, dict):
            with open(path, "w") as f:
                json.dump(written_metrics, f)
    else:
        with open(path, "w") as f:
            json.dump(metrics, f)

def get_matches(
    structure: Structure, alternatives: pd.Series, matcher: StructureMatcher
) -> tuple[list[int], list[float]]:
    with redirect_stdout(trap):
        structure = to_structure(structure)
    matches, rms_dists = [], []
    for ind, alt in alternatives.items():
        with redirect_stdout(trap):
            alt_structure = to_structure(alt)
        rms_dist = matcher.get_rms_dist(structure, alt_structure)
        if rms_dist is not None:
            rms_dist, *_ = rms_dist
            rms_dists.append(rms_dist)
            matches.append(ind)
    return matches, rms_dists


def main(args: Namespace) -> None:
    '''
    '''
    # get_hash = lambda structure: str(sorted(list(structure.get_primitive_structure().atomic_numbers)))
    get_hash = lambda structure: str(sorted(list(structure.atomic_numbers)))
    print('preprocess and load novelty reference')
    path_to_reference_folder = f'{os.getcwd()}/data/mp_20/'
    path_to_reference = os.path.join(path_to_reference_folder, 'novelty_reference.pt')
    if not os.path.exists(path_to_reference):
        path_to_train_part = os.path.join(path_to_reference_folder, 'train.csv')
        dataset = pd.read_csv(path_to_train_part, index_col=0)
        train_structures = dataset['cif'].apply(Structure.from_str, fmt="cif")
        reference = defaultdict(list)
        for structure in train_structures:
            structure_hash = get_hash(structure)
            reference[structure_hash].append(structure)
        torch.save(reference, path_to_reference)
    else:
        reference = torch.load(path_to_reference)

    print('load generated structures')
    df = pd.read_json(args.json_in)
    df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)

    print('compute stable structures')
    if args.ehulls is not None:
        df_hull = pd.read_json(args.ehulls)
        df = df.join(df_hull, how="inner")
        # filter out high energy structures
        df = df[df[args.e_above_hull_column] <= args.e_above_hull_maximum]

    df = filter_prerelaxed(
        df,
        args.num_structures,
        maximum_nary=args.maximum_nary,
        minimum_nary=args.minimum_nary - 1,
    )

    path_json_sun_count = args.json_out.parent / args.json_sun_count
    save_metrics_only_overwrite_newly_computed(
        path_json_sun_count, {"num_stable": len(df)}
    )

    matcher = StructureMatcher()  # MatterGen Novelty settings

    print('compute stable unique structures')
    # stable, unique
    stable_structure_list = [to_structure(x) for x in df['structure'].to_list()]
    present = defaultdict(list)
    uniqueness = []
    for structure in stable_structure_list:

        structure_hash = get_hash(structure)
        if structure_hash not in present:
            uniqueness.append(True)
        else:
            for present_structure in present[structure_hash]:
                if matcher.fit(structure, present_structure):
                    uniqueness.append(False)
                    break
            else:
                uniqueness.append(True)
        present[structure_hash].append(structure)
    save_metrics_only_overwrite_newly_computed(
        path_json_sun_count, {"num_SU": sum(uniqueness)}
    )
    su_structure_list = [x for x, is_unique in zip(stable_structure_list, uniqueness) if is_unique]

    print('compute SUN structures')
    novelty = []
    for structure in su_structure_list:
        
        structure_hash = get_hash(structure)
        if structure_hash not in reference:
            novelty.append(True)
        else:
            for reference_structure in reference[structure_hash]:
                if matcher.fit(structure, reference_structure):              
                    novelty.append(False)
                    break
            else:
                novelty.append(True)
    save_metrics_only_overwrite_newly_computed(
        path_json_sun_count, {"num_SUN": sum(novelty)}
    )

    # for nov, struct in zip(novelty, su_structure_list):
    #     if nov:
    #         plot_stable_struc(struct)

if __name__ == "__main__":
    '''Get unique novel rate (when passing ehulls=None).
    otherwise get stable unique novel rate (SUN)
    '''
    parser = ArgumentParser()
    parser.add_argument("json_in", type=Path, help="prerelaxed dataframe")
    parser.add_argument("json_out", type=Path, help="new dataframe")
    parser.add_argument(
        "--tabular_dataset",
        type=str,
        choices=VALID_TABULAR_DATASETS,
        default="diffcsp_mp20",
    )
    parser.add_argument("-n", "--num_structures", type=int, default=None)
    parser.add_argument("--slurm_partition", type=str, default="ocp")
    parser.add_argument(
        "--maximum_nary",
        type=int,
        default=None,  # we know there aren't structures in the dataset with more than this
        help="Any queries to structures with higher nary are avoided.",
    )
    parser.add_argument(
        "--minimum_nary",
        type=int,
        default=2,
        help="Any queries to structures with lower nary are avoided.",
    )
    parser.add_argument("--ehulls", type=str, default=None)
    parser.add_argument(
        "--e_above_hull_column", type=str, default="e_above_hull_per_atom_dft_corrected"
    )
    parser.add_argument("--e_above_hull_maximum", type=float, default=0) # meta stable (0.1eV/atom), stable (0meV/atom)
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument("--json_sun_count", type=str, default="sun_count.json")
    args = parser.parse_args()

    main(args)
