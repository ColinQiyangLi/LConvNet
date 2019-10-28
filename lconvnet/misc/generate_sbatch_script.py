import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating sbatch script for slurm")
    parser.add_argument("--dir", help="the path to the folder of interest")
    parser.add_argument(
        "--partition", "-p", help="partition on the slurm", default="p100"
    )
    parser.add_argument(
        "--cpu", "-c", help="# of cpus", default=2
    )
    parser.add_argument(
        "--mem", help="amount of memory", default="10G"
    )
    parser.add_argument(
        "--gres", help="gres", default="gpu:1"
    )
    parser.add_argument(
        "--limit", help="limit on number of jobs", default=12
    )
    parser.add_argument("--pattern",
                        help="find all the file that ends with this in the folder",
                        default="cfg.yaml")
    parser.add_argument("--out",
                        help="output directory relative to dir",
                        default="batch_run.sh")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    lst = []
    print("=========== Searching for {} ==================".format(args.pattern))
    for dir_name, subdir_list, file_list in os.walk(args.dir):
        for fname in file_list:
            if args.pattern == fname:
                fn = os.path.join(dir_name, fname)
                print('  %s' % (fn))
                lst.append("\"python -m lconvnet.run --cfg " + fn + (" --resume" if args.resume else "") + (" --test" if args.test else "") + "\"")
    print("Found {} experiments".format(len(lst)))

    print("Generating sbatch script....")
    out_dir = os.path.join(args.dir, args.out)
    print(out_dir)
    with open(out_dir, "w") as f:
        f.write("""#!/bin/bash
#SBATCH --partition={}
#SBATCH --gres={}
#SBATCH --mem={}
#SBATCH --array=0-{}%{}
#SBATCH -c {}

list=(
    {}
)
echo "Starting task $SLURM_ARRAY_TASK_ID: ${{list[SLURM_ARRAY_TASK_ID]}}"
eval ${{list[SLURM_ARRAY_TASK_ID]}}\n""".format(args.partition, args.gres, args.mem, len(lst) - 1, args.limit, args.cpu, "\n    ".join(lst)))
    print("Done.")
