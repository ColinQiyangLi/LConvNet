import argparse
import os
from shutil import copyfile

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating sbatch script for slurm")
    parser.add_argument("--dir", help="the path to the folder of interest")

    args = parser.parse_args()

    for dir_name, _, file_list in os.walk(args.dir):
        if any(map(lambda x: "results" in x, file_list)):
            rel_path = os.path.relpath(dir_name, args.dir)
            fn = list(sorted(filter(lambda x: "results" in x, file_list)))[-1]
            res_file = os.path.join(args.dir, rel_path, "results.yaml")
            assert not os.path.exists(res_file)
            copyfile(os.path.join(args.dir, rel_path, fn), res_file)
