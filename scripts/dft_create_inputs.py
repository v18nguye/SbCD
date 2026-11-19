from __future__ import annotations
"""Copyright (c) Meta Platforms, Inc. and affiliates."""

"""
There are several relevant settings for DFT.
1. The MP20 settings. This is what we use to evaluate CDVAE and DiffCSP which were trained on that data.

2. The OCP settings. These will be used to evaluate models trained on larger datasets than MP20.
"""
import io

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml
from joblib import Parallel, delayed
from pymatgen.core import Structure
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.io.vasp.sets import MPRelaxSet
from tqdm import tqdm
from contextlib import redirect_stdout

from pymatgen.core import (
    Composition,
    DummySpecies,
    Element,
    Lattice,
    Species,
    Structure,
)

trap = io.StringIO()

CHGNET_STABILITY_THRESHOLD = 0.1

COLUMNS_COMPUTATIONS = {
    "composition": lambda df: df[f"structure"].map(get_composition_dict),
    "chemsys": lambda df: df["composition"].map(get_chemsys),
    "nary": lambda df: df["chemsys"].map(get_nary),
}

def to_structure(structure: Structure | dict | str) -> Structure:
    with redirect_stdout(trap):
        if isinstance(structure, dict):
            return Structure.from_dict(structure)
        elif isinstance(structure, str):
            return Structure.from_str(structure, fmt="cif")
        else:
            return structure

def get_nary(
    chemsys: set[Element, Species, DummySpecies] | list[Element, Species, DummySpecies]
) -> int:
    return len(set(chemsys))


def get_composition(
    structure: Structure | dict,
) -> Composition:
    if structure == {}:
        return Composition()
    return to_structure(structure).composition


def get_composition_dict(
    structure: Structure | dict,
) -> dict[str, float]:
    return get_composition(structure).as_dict()


def get_chemsys(
    composition: Composition | dict,
) -> set[Element | Species | DummySpecies] | None:
    elements = list(elt.name for elt in to_composition(composition).elements)
    return set(elements)

def to_composition(composition: Composition | dict) -> Composition:
    if isinstance(composition, dict):
        return Composition.from_dict(composition)
    else:
        return composition

def maybe_get_missing_columns(
    df: pd.DataFrame, maps: dict[str, callable | dict]
) -> pd.DataFrame:
    for name, mapping in maps.items():
        if name not in df.columns:
            df[name] = mapping(df)
    return df

def filter_prerelaxed(
    df: pd.DataFrame,
    num_structures: int | None = None,
    filter_exceptions: bool = True,
    filter_converged: bool = False,
    # maxima: dict[str, float | int] = {"e_delta": 15.0},
    maxima: dict[str, float | int] = {},
    maximum_nary: int | None = None,
    minimum_nary: int = 0,
) -> pd.DataFrame:
    if filter_exceptions:
        df = df[df["exception"] == False]

    if filter_converged:
        df = df[df["converged"] == True]
    for key, value in maxima.items():
        df = df[df[key] < value]

    if maximum_nary is not None:
        df = df[df["chemsys"].map(len) <= maximum_nary]
    df = df[df["chemsys"].map(len) > minimum_nary]

    if num_structures is not None:
        print(f"limiting to the first {num_structures} samples after filtering")
        df = df.iloc[:num_structures, :]

    return df

def get_intersection(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.Series(list(set(a) & set(b)))

def get_chemsys(
    composition: Composition | dict,
) -> set[Element | Species | DummySpecies] | None:
    elements = list(elt.name for elt in to_composition(composition).elements)
    return set(elements)

def parse_vasp(path: Path | str):
    path = Path(path)
    vrun = Vasprun(filename=path)
    vrun.final_structure
    vrun.final_energy


def write_vasp_directory(
    structure: Structure | dict,
    path: Path | str,
    potcar_spec: bool = False,
) -> None:
    path = Path(path)
    structure = to_structure(structure)
    relax_set = MPRelaxSet(structure=structure)
    try:
        relax_set.write_input(
            output_dir=str(path.resolve()),
            make_dir_if_not_present=True,
            potcar_spec=potcar_spec,
        )
    except (TypeError, KeyError) as exp:
        print(f"{exp}")
    return None


def subsample_by_nary(
    df: pd.DataFrame, num_structures: int
) -> tuple[pd.DataFrame, dict[int, float]]:
    assert num_structures <= len(df)
    weight_map = (df.groupby("nary").count()["e_relax"] / len(df)).to_dict()
    df["weights"] = df["nary"].map(weight_map)
    return df.sample(num_structures, weights=df["weights"]), weight_map


def subsample(df: pd.DataFrame, num_structures: int) -> pd.DataFrame:
    return df.sample(num_structures)


def get_equal_weights(df: pd.DataFrame) -> dict[int, float]:
    return {i: 1 / df["nary"].max() for i in range(1, df["nary"].max() + 1)}


def main(args: Namespace) -> None:
    print(f"{args.root=}")

    # easier to reason with positives...
    filter_converged = not args.do_not_filter_converged

    # load data
    print("reading json")
    df = pd.read_json(args.json_in)
    print(f"{len(df)=} loaded")
    df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)
    print(f"{len(df)=} get missing cols")
    df = filter_prerelaxed(
        df,
        filter_converged=filter_converged,
        maximum_nary=args.maximum_nary,
    )
    print(f"{len(df)=} filtered prerelaxed")
    # no potential for these
    df = df[~df["chemsys"].map(lambda x: "Ra" in x)]
    df = df[~df["chemsys"].map(lambda x: "At" in x)]
    df = df[~df["chemsys"].map(lambda x: "Po" in x)]
    print(f"{len(df)=} removed structures without labels")
    df["nary"] = df["chemsys"].map(get_nary)

    if args.ehulls is not None:
        df_hull = pd.read_json(args.ehulls)
        df = df.join(df_hull, how="inner")
        # filter out high energy structures
        df = df[df["e_above_hull_per_atom_chgnet"] < CHGNET_STABILITY_THRESHOLD]

    print("finished")

    num_structures = min(args.max_num_structures, len(df))
    print(f"In the end, producing {num_structures=} dft input files")

    # choose subsampling stragegy
    if args.subsample is None:
        pass
    elif args.subsample == "first":
        df = df[:num_structures]
    elif args.subsample == "random":
        print("subsample randomly with equal weight")
        df = subsample(df, num_structures)
        weights = get_equal_weights(df)
    elif args.subsample == "nary":
        print("subsample weighted by nary")
        df, weights = subsample_by_nary(df, num_structures)
    else:
        raise ValueError()
    print(f"{len(df)=}")

    # save basics
    args.root.mkdir(parents=True, exist_ok=True)
    with open(args.root / "args.yaml", "w") as f:
        to_save = deepcopy(vars(args))
        to_save["json_in"] = str(to_save["json_in"])
        to_save["root"] = str(to_save["root"])
        yaml.dump(to_save, f)
    with open(args.root / "weights.yaml", "w") as f:
        yaml.dump(weights, f)
    df.drop("structure", axis=1).to_csv(args.root / "dataframe.csv")

    Parallel(n_jobs=args.n_jobs)(
        delayed(write_vasp_directory)(structure, args.root / f"{i:06d}")
        for i, structure in tqdm(df["structure"].items())
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("json_in", type=Path, help="prerelaxed dataframe")
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--subsample",
        type=str,
        choices=["first", "random", "nary"],
        default="random",
    )
    parser.add_argument("--max_num_structures", type=int, default=10_000)
    parser.add_argument(
        "--maximum_nary",
        type=int,
        default=None,
        help="Any queries to structures with higher nary are avoided.",
    )
    parser.add_argument(
        "--ehulls",
        type=Path,
        default=None,
        help="if given, filter to chgnet e_above_hull_per_atom < 0.01",
    )  # an alternative is to triage with chgnet instead of cutoff
    parser.add_argument("--do_not_filter_converged", action="store_true")
    parser.add_argument("--n_jobs", type=int, default=1)
    args = parser.parse_args()

    main(args)
