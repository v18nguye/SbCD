"""Copyright (c) Meta Platforms, Inc. and affiliates."""

from __future__ import annotations

import os, time
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import submitit
import torch
from chgnet.model import CHGNet
from chgnet.model import StructOptimizer

from pymatgen.core import (
    Lattice,
    Structure,
)
import multiprocessing

import sys
sys.path.append('.')
from scripts.chgnet_ import RelaxationDataGHGNET, prerelax_with_chgnet
import pdb

def get_available_cpus():
    return int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))

def compute_safe_parallelism(n_jobs, cpus_per_task):
    return max(1, min(n_jobs, get_available_cpus() // cpus_per_task))
#
# Here's a really nice script for CHGNet
# https://github.com/janosh/matbench-discovery/blob/main/models/chgnet/test_chgnet.py
#

def get_custom_structure(gen: dict, i: int, max_atoms=72) -> Structure:
    indexes = torch.cumsum(gen["num_atoms"].squeeze(), dim=0)
    start = 0 if i == 0 else int(indexes[i - 1])
    end = int(indexes[i])

    lengths = gen["lengths"][i]
    angles = gen["angles"][i]
    atom_types = gen["atom_types"][start:end] # TODO: check this.
    atom_types =  torch.argmax(atom_types, axis=-1) + 1
    frac_coords = gen["frac_coords"][start:end]
    print(f'Num unitcell atoms ={len(frac_coords)}')
    is_many_atoms = len(frac_coords) > max_atoms
    return Structure(
        lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
        species=atom_types,
        coords=frac_coords,
        coords_are_cartesian=False,
    ), is_many_atoms 

def wait_for_jobs_to_finish(jobs: list, sleep_time_s: int = 5) -> None:
    # wait for the job to be finished
    num_finished = 0
    print(f"number of jobs: {len(jobs):02d}")
    while num_finished < len(jobs):
        time.sleep(sleep_time_s)
        num_finished = sum(job.done() for job in jobs)
        print(f"number finished: {num_finished:02d}", flush=True, end="\r")
    print("")
    print("jobs done!")
    return None


def prerelax_multiple(
    crystals,
    device,
    steps: int,
    index: np.ndarray,
    json_file: Path,
    max_atoms=200,
) -> pd.DataFrame:
    """cdvae structures, max relaxation steps, list of indexes -> json file"""
    chgnet = CHGNet.load(use_device=device)
    relaxer = StructOptimizer(model=chgnet, use_device=device)
    rd = RelaxationDataGHGNET()
    for i in index.tolist():
        try:
            message = ''
            is_valid = True
            crys = crystals[i]
            num_atoms = len(crys.frac_coords)
            print(f'num_atoms: {num_atoms}')
            if num_atoms > max_atoms:
                is_valid=False
                message += 'many_atoms'
            if not crys.comp_valid: # comp_valid (10k)| valid (1k)
                is_valid=False
                message +='+invalid_crys'

            if is_valid:
                structure = crys.structure
                pair = prerelax_with_chgnet(crys.structure, chgnet, relaxer, steps)
                rd.index.append(i)
                rd.e_gen.append(pair.energies[0])
                rd.e_relax.append(pair.energies[1]) # see 'Energy' column in chgnet_log_dir/*.json 
                rd.n_to_relax.append(pair.n_steps_to_relax)
                rd.rms_dist.append(pair.rms_dist)
                rd.matched.append(pair.match)
                rd.converged.append(True if pair.n_steps_to_relax < steps else False)
                rd.exception.append(False)
                rd.num_sites.append(structure.num_sites)
                rd.structure.append(pair.structure_dicts[0])
            else:
                print(f'errors: {message}')
                rd.index.append(i)
                rd.e_gen.append(f'errors:{message}') # can not run on many atoms (memory extreme expensive)
                rd.e_relax.append(None)
                rd.n_to_relax.append(pd.NA)
                rd.rms_dist.append(None)
                rd.matched.append(False)
                rd.converged.append(False)
                rd.exception.append(True)
                rd.num_sites.append(pd.NA)
                rd.structure.append({})
        except Exception as exp:
            print(exp)
            rd.index.append(i)
            rd.e_gen.append(None)
            rd.e_relax.append(None)
            rd.n_to_relax.append(pd.NA)
            rd.rms_dist.append(None)
            rd.matched.append(False)
            rd.converged.append(False)
            rd.exception.append(True)
            rd.num_sites.append(pd.NA)
            rd.structure.append({})
    # dataclasses.fields(rd) produces blank :(
    # dataclasses.asdict(rd) also produces blank
    pre_pandas = dict(
        (field.name, getattr(rd, field.name))
        for field in getattr(rd, "__dataclass_fields__").values()
    )
    df = pd.DataFrame.from_dict(pre_pandas).set_index("index")
    df.loc[:, "e_delta"] = np.abs(df.loc[:, "e_relax"] - df.loc[:, "e_gen"])
    df.to_json(json_file)
    return df

def main(args: Namespace) -> None:
    # get file
    device = 'cuda:0'
    start = time.time()
    gen_file = Path(args.path_to_structures)
    print(f"reading file: {str(gen_file.resolve())}")
    # determine how many structures there are
    structures = torch.load(gen_file, map_location=device)
    if args.num_structures is not None:
        assert args.num_structures <= len(structures)
        num_structures = args.num_structures
    else:
        num_structures = len(structures)
    print(f"{num_structures=}")
    # split structures into parts
    index = np.arange(num_structures)
    indexes = np.array_split(index, args.num_jobs)
    indexes = [i for i in indexes if i.size > 0]  # filter out blanks

    # compute the energies
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir) 

    files = [
        Path(args.log_dir) / f"{index.min():07d}-{index.max():07d}.json"
        for index in indexes
    ]
    # prerelax_multiple(structures, device, args.steps, indexes[0], files[0]) # for debugging
    # set cluster
    cluster = "local" if args.slurm_partition is None else "slurm"
    if args.debug:
        cluster = "debug"

    executor = submitit.AutoExecutor(
        folder=args.log_dir,
        cluster=cluster,
    )
    cpus_per_task=get_available_cpus()//args.num_jobs
    safe_jobs = compute_safe_parallelism(args.num_jobs, cpus_per_task)
    executor.update_parameters(
        slurm_array_parallelism=safe_jobs,
        nodes=1,
        slurm_ntasks_per_node=1,
        cpus_per_task=cpus_per_task,
        gpus_per_node=1,
        timeout_min=args.timeout_min,
        slurm_partition=args.slurm_partition,
    )
    doit = partial(prerelax_multiple, structures, device, args.steps)
    jobs = executor.map_array(doit, indexes, files)

    wait_for_jobs_to_finish(jobs)
    # write to file
    df = pd.concat([job.result() for job in jobs])
    df.to_json(Path(args.path_to_json))
    print(f"wrote file to: ")
    print(f"{args.path_to_json}")
    end = time.time()
    print(f"Execution time: {end-start}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "path_to_structures",
        type=Path,
        help="path to the eval_pt, typically `eval_for_dft.pt`, file from cdvae / diffcsp / rfm",
    )
    parser.add_argument("path_to_json", type=Path, help="output")
    parser.add_argument("log_dir", type=Path)
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=1,
        help="number of jobs to divide structures between, only works when --slurm_partition is set",
    )
    parser.add_argument("-n", "--num_structures", type=int, default=None)
    parser.add_argument(
        "--steps", type=int, default=1_500, help="maximum number of steps in relaxation"
    )  
    parser.add_argument("--timeout_min", type=int, default=300000) 
    parser.add_argument("--slurm_partition", type=str, default=None) 
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run the prerelaxations sequentially in the same process as this script.",
    )
    args = parser.parse_args()

    main(args)