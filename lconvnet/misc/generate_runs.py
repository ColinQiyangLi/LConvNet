import argparse
import os
from oyaml import load, dump
from collections import defaultdict, OrderedDict
from itertools import product


def walk(s, choose_one=False):
    if choose_one:
        assert isinstance(s, dict)
        m = []
        for key, value in s.items():
            m.extend([(os.path.join(key, str(a)), b) for (a, b) in walk(value)])
        return m

    choose_one_flag = set()
    if isinstance(s, list):
        its = list(enumerate(s))
    elif isinstance(s, dict):
        s_new = OrderedDict()
        for key, value in s.items():
            if "|" in key:
                rk, rid = key.split("|")
                if rk not in s_new:
                    s_new[rk] = OrderedDict()
                s_new[rk][rid] = value
                choose_one_flag.add(rk)
            else:
                s_new[key] = value
        its = list(filter(lambda x: "|" not in x[0], s_new.items()))
    else:
        return [("", s)]
    m = [list(walk(value, key in choose_one_flag)) for key, value in its]
    m = list(product(*m))

    keys = list(map(lambda x: x[0], its))
    if isinstance(s, list):
        res = [(os.path.join(*[pp[0] for pp in p]), [pp[1] for pp in p]) for p in m]
    if isinstance(s, dict):
        res = [(os.path.join(*[pp[0] for pp in p]), OrderedDict((key, pp[1])
                                                                for key, pp in zip(keys, p))) for p in m]
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generating configs for experiments")
    parser.add_argument("--dir", help="the path to the template of interest")
    parser.add_argument(
        "--template", help="the name for the template yaml", default="template.yaml")
    parser.add_argument(
        "--pattern", help="the name for the cfg yaml", default="cfg.yaml")
    args = parser.parse_args()

    out_dir = args.dir
    with open(os.path.join(out_dir, "template.yaml"), "r") as f:
        s = load(f)

    lst = walk(s)
    print("Found {} combinations".format(len(lst)))
    for index, (a, b) in enumerate(lst):
        print("Processing run-{}: \t {}".format(index, a))
        path_dir = os.path.join(out_dir, a)
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        with open(os.path.join(path_dir, args.pattern), "w") as f:
            dump(b, f)
    print("Done.")
