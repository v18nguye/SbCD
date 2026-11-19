#!/bin/bash 

export CUDA_VISIBLE_DEVICES=0
export PMG_VASP_PSP_DIR=PATH_TO_YOUR_PSEUDOPOTENTIALS
export num_jobs=1
export num_structures=10000

work_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || exit 1
export gt_file="$work_dir/data/mpts_52/test.csv"
export gt_crys_file="$work_dir/data/mpts_52/test_gt_crys.pt"
export PROJECT_ROOT=$work_dir
export HYDRA_JOBS=$work_dir
export WABDB_DIR=$work_dir
export WABDB_CACHE_DIR=$work_dir


while getopts t:e: flag
do
    case "${flag}" in
        t) task=${OPTARG};;
        e) exp=${OPTARG};; # for metric, sampling, it's a path to saved model; others just name exp.
    esac
done

relax_mlip=chgnet
data=mpts52

echo "run task: $task, data: $data, exp: $exp";

if [ "$task" = "train" ]; then
    python sbcd/run.py data=$data model=$data expname=$exp

elif [ "$task" = "metric" ]; then
    python scripts/compute_metrics.py --root_path $exp --tasks gen --gt_file $gt_file --gt_crys_file $gt_crys_file

elif [ "$task" = "sampling" ]; then
    python scripts/generation.py --model_path $exp

elif [ "$task" = "prelax" ]; then # chgnet pre-relaxation
    python scripts/prerelax.py "$exp/gen_crys_.pt" "$exp/${relax_mlip}_eval_gen_prelaxed.json" "$exp/${relax_mlip}_log_dir" --num_jobs $num_jobs --num_structures $num_structures

elif [ "$task" = "ehull" ]; then
    python scripts/ehull.py "$exp/${relax_mlip}_eval_gen_prelaxed.json" "$exp/${relax_mlip}_ehulls.json"
    
elif [ "$task" = "sun" ]; then
    python scripts/novelty.py "$exp/${relax_mlip}_eval_gen_prelaxed.json" "$exp/${relax_mlip}_sun.json" --ehulls "$exp/${relax_mlip}_ehulls.json" --e_above_hull_column "e_above_hull_per_atom_chgnet" --json_sun_count "$exp/${relax_mlip}_sun_count.json" # or "$exp/ehulls_corrected.json" most accurate

else
    echo "no task to run !!!"
fi