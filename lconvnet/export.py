from collections import defaultdict
import argparse
import yaml
import os
import numpy as np

from datetime import datetime

from pylatex import (
    Document,
    Tabularx,
    Tabu,
    Center,
    MultiColumn,
    MultiRow,
    Math,
    Quantity,
    NoEscape,
    UnsafeCommand,
    MiniPage,
    Section,
    Package,
)

CIFAR10Repr = "\\textbf{{CIFAR10}}"
MNISTRepr = "\\textbf{{MNIST}}"
WDEReprSTL10 = "\\textbf{Wasserstein Distance (STL-10)}"
WDEReprCIFAR10 = "\\textbf{Wasserstein Distance (CIFAR-10)}"


def parse_path(path, config, mode, eps):
    method_str_patterns = [
        ("OSSN", "OSSN"),
        ("RKO", "RKO"),
        ("SVCM", "SVCM"),
        ("BCOP-Bjorck", "BCOP"),
        ("BCOP/", "BCOP"),
        ("L2Nonexpansive", "RK-L2NE"),
        ("BCOP-Fixed", "BCOP-Fixed"),
        ("kw", "KW"),
        ("fc", "FC"),
        ("qian", "QW"),
    ]
    dataset_str_patterns = [
        ("-cifar10", CIFAR10Repr),
        ("-mnist", MNISTRepr),
        ("wde_cifar10", WDEReprCIFAR10),
        ("wde_stl10", WDEReprSTL10),
    ]
    arch_str_patterns = [
        ("small", "Small"),
        ("large", "Large"),
        ("3-layer", "3"),
        ("model-3", "3"),
        ("model-4", "4"),
        ("maxmin", "MaxMin"),
        ("relu", "ReLU"),
    ]
    config_str_patterns = [
        ("natural", "Clean", "Clean (*)"),
        ("lb", "Robust", "Robust (*)"),
        ("boundary_attack-ub", "BA", "BA (*)"),
        ("pointwise-ub", "PA", "PA (*)"),
        ("pgd-ub", "PGD", "PGD (*)"),
        ("fgsm-ub", "FGSM", "FGSM (*)"),
        ("loss", "Loss", "Loss (*)"),
    ]
    lr_patterns = [
        ("lr-0.0001", "lr-0.0001"),
        ("lr-0.001", "lr-0.001"),
        ("lr-0.01", "lr-0.01"),
        ("lr-0.1", "lr-0.1"),
    ]

    names = [None, None, None, None, None, eps]
    for index, patterns in enumerate(
        [dataset_str_patterns, arch_str_patterns, method_str_patterns, lr_patterns]
    ):
        for p in patterns:
            if p[0] in path:
                names[index] = p[1]
    for p in config_str_patterns:
        if p[0] in config:
            if mode == 0:
                names[4] = p[2]
            else:
                names[4] = p[1]
    return tuple(names)


def data_export(
    d,
    f_keys_0=["loss"],
    f_keys_1=["natural", "lb", "fgsm-ub", "pgd-ub"],
    f_keys_2=["natural", "pointwise-ub", "boundary_attack-ub"],
):
    results = defaultdict(list)
    for dir_name, _, file_list in os.walk(d):
        fn = "results.yaml"
        if not fn in file_list:
            continue
        rel_path = os.path.relpath(dir_name, d)
        with open(os.path.join(d, rel_path, fn)) as f:
            res = yaml.safe_load(f)
            for f_key in f_keys_0:
                if f_key in res.keys():
                    tup = parse_path(rel_path, f_key, None, None)
                    results[tup].append(res[f_key])
                    print(rel_path, tup)
            search_space = []
            if "adv_robustness_mini_eval" in res:
                search_space.append((res["adv_robustness_mini_eval"], f_keys_2, 0))
            if "adv_robustness" in res:
                search_space.append((res["adv_robustness"], f_keys_1, 1))
            for source, f_keys, index in search_space:
                for eps, val in source.items():
                    eps = str(eps)
                    for f_key in f_keys:
                        if f_key not in val:
                            continue
                        tup = parse_path(rel_path, f_key, index, eps)
                        results[tup].append(val[f_key])

    final_results = defaultdict(lambda: None)

    for key, value in results.items():
        final_results[key] = {}
        if key[4] == "Loss":
            final_results[key]["mean"] = -np.mean(value)
            final_results[key]["std"] = np.std(value) if len(value) > 1 else None
            final_results[key]["num"] = len(value)
        else:
            final_results[key]["mean"] = np.mean(value) * 100.0
            final_results[key]["std"] = (
                np.std(value) * 100.0 if len(value) > 1 else None
            )
            final_results[key]["num"] = len(value)

    # print(final_results)
    geometry_options = {
        "landscape": True,
        "margin": "1.5in",
        "headheight": "20pt",
        "headsep": "10pt",
        "includeheadfoot": True,
    }
    doc = Document(page_numbers=True, geometry_options=geometry_options)
    doc.packages.append(Package("booktabs"))
    with doc.create(Section("Table 1", numbering=False)):
        get_latex_table_type1(doc, final_results)

    with doc.create(Section("Table 2", numbering=False)):
        get_latex_table_type1b(doc, final_results)

    with doc.create(Section("Table 3", numbering=False)):
        with doc.create(MiniPage(width=r"0.5\textwidth")):
            get_latex_table_type2(
                doc, final_results, config_names=["Clean", "PGD", "FGSM"]
            )
        with doc.create(MiniPage(width=r"0.5\textwidth")):
            get_latex_table_type2(
                doc, final_results, config_names=["Clean (*)", "BA (*)", "PA (*)"]
            )

    with doc.create(Section("Table 4a-Maxmin-lr-0.0001", numbering=False)):
        get_latex_table_type3(
            doc,
            final_results,
            dataset_name=WDEReprSTL10,
            arch_names=["MaxMin"],
            lr="lr-0.0001",
        )
    with doc.create(Section("Table 4a-ReLU-lr-0.001", numbering=False)):
        get_latex_table_type3(
            doc,
            final_results,
            dataset_name=WDEReprSTL10,
            arch_names=["ReLU"],
            lr="lr-0.001",
        )
    with doc.create(Section("Table 4b-Maxmin-lr-0.0001", numbering=False)):
        get_latex_table_type3(
            doc,
            final_results,
            dataset_name=WDEReprCIFAR10,
            arch_names=["MaxMin"],
            lr="lr-0.0001",
        )
    with doc.create(Section("Table 4b-ReLU-lr-0.001", numbering=False)):
        get_latex_table_type3(
            doc,
            final_results,
            dataset_name=WDEReprCIFAR10,
            arch_names=["ReLU"],
            lr="lr-0.001",
        )
    with doc.create(Section("Table 7", numbering=False)):
        get_latex_table_type1(
            doc, final_results, method_names=["BCOP-Fixed", "RK-L2NE", "BCOP"]
        )

    with doc.create(Section("Table 8", numbering=False)):
        get_latex_table_type1b(
            doc,
            final_results,
            method_arch_names=[("BCOP", "Large"), ("FC", "3"), ("MMR", "Universal")],
            eps=["0.3", "0.1"],
        )

    with doc.create(Section("Table 9", numbering=False)):
        get_latex_table_type1b(
            doc,
            final_results,
            method_arch_names=[("BCOP", "Large"), ("FC", "3"), ("QW", "3"), ("QW", "4")],
            config_names=["Clean", "Robust", "PGD", "FGSM"],
        )
    return doc


def cmidrule(table, start, end):
    table.append(UnsafeCommand("cmidrule", "{}-{}".format(start, end)))


def boldify(s, bold=True, math=False):
    if not bold:
        return "{:.2f}".format(s) if math else s
    if math:
        return "\\mathbf{{{:.2f}}}".format(s)
    return "\\textbf{{{}}}".format(s)


def populate_cell(entry, bold=False):
    if entry is None:
        return "-"
    assert entry["num"] == 1 or entry["num"] == 5, entry["num"]
    if entry["std"] is None:
        cell = NoEscape("${}$".format(boldify(entry["mean"], bold=bold, math=True)))
    else:
        cell = NoEscape(
            "${}\\pm{}$".format(
                boldify(entry["mean"], bold=bold, math=True),
                boldify(entry["std"], bold=False, math=True),
            )
        )
    return cell


def populate_numbers(entries):
    best_index = max(
        range(len(entries)),
        key=lambda x: 0.0 if entries[x] is None else entries[x]["mean"],
    )
    cells = ()
    for cell in [
        (populate_cell(entry, best_index == index),)
        for index, entry in enumerate(entries)
    ]:
        cells += cell
    return cells


def get_latex_table_type1(
    doc,
    results=None,
    method_names=["OSSN", "RKO", "SVCM", "BCOP"],
    config_names=["Clean", "Robust"],
    eps=["1.58", "0.1411764706"],
):

    dataset_names = [MNISTRepr, CIFAR10Repr]
    arch_names = ["Small", "Large"]
    num_archs = len(arch_names)
    num_configs = len(config_names)
    num_methods = len(method_names)
    fmt = "c|cc|" + "c" * num_methods
    num_cols = 3 + num_methods
    with doc.create(Tabularx(fmt)) as data_table:
        headers = tuple(
            map(
                lambda x: NoEscape(boldify(x)),
                ("Dataset", "", "") + tuple(method_names),
            )
        )
        data_table.add_row(headers)
        cmidrule(data_table, 1, num_cols)
        for dataset_name, ep in zip(dataset_names, eps):
            multi_row_outer = MultiRow(
                num_archs * num_configs, data=NoEscape(dataset_name)
            )
            for i, arch_name in enumerate(arch_names):
                multi_row_inner = MultiRow(
                    num_configs, data=NoEscape(boldify(arch_name))
                )
                for j, config_name in enumerate(config_names):

                    row = ()
                    if i == 0 and j == 0:
                        row += (multi_row_outer,)
                    else:
                        row += ("",)
                    if j == 0:
                        row += (multi_row_inner,)
                    else:
                        row += ("",)
                    row += (config_name,)
                    row += populate_numbers(
                        [
                            results[
                                (
                                    dataset_name,
                                    arch_name,
                                    method_name,
                                    None,
                                    config_name,
                                    ep,
                                )
                            ]
                            for method_name in method_names
                        ]
                    )
                    data_table.add_row(row)
                if i != num_archs - 1:
                    cmidrule(data_table, 2, num_cols)
            cmidrule(data_table, 1, num_cols)


def get_latex_table_type1b(
    doc,
    results=None,
    method_arch_names=[
        ("BCOP", "Large"),
        ("FC", "3"),
        ("KW", "Large "),
        ("KW", "Resnet"),
    ],
    eps=["1.58", "0.1411764706"],
    config_names=["Clean", "Robust"],
):
    dataset_names = [MNISTRepr, CIFAR10Repr]
    num_configs = len(config_names)
    num_methods = len(method_arch_names)
    fmt = "c|c|" + "c" * num_methods
    num_cols = 2 + num_methods
    with doc.create(Tabularx(fmt)) as data_table:
        headers = (NoEscape(boldify("Dataset")), "") + tuple(
            map(lambda x: NoEscape(boldify("{}-{}".format(*x))), method_arch_names)
        )
        data_table.add_row(headers)
        cmidrule(data_table, 1, num_cols)
        for dataset_name, ep in zip(dataset_names, eps):
            multi_row_outer = MultiRow(num_configs, data=NoEscape(dataset_name))
            for j, config_name in enumerate(config_names):
                row = ()
                if j == 0:
                    row += (multi_row_outer,)
                else:
                    row += ("",)
                row += (config_name,)
                row += populate_numbers(
                    [
                        results[
                            (
                                dataset_name,
                                arch_name,
                                method_name,
                                None,
                                config_name,
                                ep,
                            )
                        ]
                        for method_name, arch_name in method_arch_names
                    ]
                )
                data_table.add_row(row)
            cmidrule(data_table, 1, num_cols)


def get_latex_table_type2(
    doc,
    results=None,
    dataset_name=CIFAR10Repr,
    config_names=["Clean", "PGD", "FGSM"],
    eps="0.1411764706",
):
    arch_names = ["Small", "Large"]
    method_names = ["KW", "BCOP"]
    num_archs = len(arch_names)
    num_methods = len(method_names)
    fmt = "l" + "c" * num_methods
    num_cols = 1 + num_methods
    with doc.create(Tabularx(fmt)) as data_table:
        headers = tuple(
            map(lambda x: NoEscape(boldify(x)), ("",) + tuple(method_names))
        )
        data_table.add_row(headers)
        cmidrule(data_table, 1, num_cols)
        for i, arch_name in enumerate(arch_names):
            data_table.add_row((arch_name,) + ("",) * num_methods)
            data_table.append(UnsafeCommand("cmidrule", "{}-{}".format(1, num_cols)))
            for j, config_name in enumerate(config_names):
                row = (config_name,)
                row += populate_numbers(
                    [
                        results[
                            (
                                dataset_name,
                                arch_name,
                                method_name,
                                None,
                                config_name,
                                eps,
                            )
                        ]
                        for method_name in method_names
                    ]
                )
                data_table.add_row(row)
            if i != num_archs - 1:
                cmidrule(data_table, 1, num_cols)


def get_latex_table_type3(
    doc,
    results=None,
    dataset_name=WDEReprSTL10,
    arch_names=["MaxMin", "ReLU"],
    lr="0.0001",
):
    method_names = ["OSSN", "RKO", "BCOP"]
    num_archs = len(arch_names)
    num_methods = len(method_names)
    fmt = "cc" + "c" * num_methods
    num_cols = 2 + num_methods
    with doc.create(Tabularx(fmt)) as data_table:
        headers = tuple(
            map(lambda x: NoEscape(boldify(x)), ("", "") + tuple(method_names))
        )
        data_table.add_row(headers)
        cmidrule(data_table, 1, num_cols)
        multi_row_outer = MultiRow(num_archs, data=NoEscape(dataset_name))
        for i, arch_name in enumerate(arch_names):
            if i == 0:
                row = (multi_row_outer,)
            else:
                row = ("",)
            row += (NoEscape(boldify(arch_name)),)
            row += populate_numbers(
                [
                    results[(dataset_name, arch_name, method_name, lr, "Loss", None)]
                    for method_name in method_names
                ]
            )
            data_table.add_row(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generating sbatch script for slurm")
    parser.add_argument("--dir", help="the path to the folder of interest")

    args = parser.parse_args()

    doc = data_export(args.dir)
    doc.generate_tex(os.path.join(args.dir, "table"))
