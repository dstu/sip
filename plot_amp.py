#!/usr/bin/env python3
#
# Example usage:
#
# plot_amp.py --help
# plot_amp.py ../amp_abstract_data -t 'ae->aA' -m SIP-ISL -o single.png
# plot_amp.py ../amp_abstract_data -t all -m SIP-ISL SIP-FST -o grid.png
# plot_amp.py ../amp_abstract_data -t all -m all --multi overlay -o overlay.png

import argparse
import functools
import glob
import os.path
import re
from typing import List, Literal

import pandas as pd
import seaborn as sns

sns.set(font_scale=1.4)

NL_NAMES = {
    "german" : "german-syll",
    "polish" : "polish",
    "turkish" : "turkish-childes",
    "english" : "english2",
    "finnish" : "finnish"
}

EXPERIMENTS = [
    "ivv",
    "harmony",
    "impossible"
    ] + list(NL_NAMES.keys())

DATA_PATHS = None
TASK_NAMES = None
EXPERIMENT_HEADER = None

def set_experiment_paths(experiment):
    global DATA_PATHS
    global TASK_NAMES
    global EXPERIMENT_HEADER

    if experiment == "ivv":
        EXPERIMENT_HEADER = "ivv"
        TASK_NAMES = {
            1: "da->Da",
            2: "da->DDa",
            3: "ad->aD",
            4: "ad->aDD",
            5: "ada->aDa",
            6: "ada->aDDa",
        }

    elif experiment == "harmony":
        EXPERIMENT_HEADER = "rebalance_harmony"
        # Maps from task index (an arbitrary number that takes the place of the glob in
        # DATA_PATHS) to a human-readable name.
        TASK_NAMES = {
            1: "ae->aA",
            2: "a.e->a.A",
            3: "a..e->a..A",
            4: "a.?e->a.?A",
            5: "a.*e->a.*A",
        }
    elif experiment == "impossible":
        EXPERIMENT_HEADER = "impossible"

        # Maps from task index (an arbitrary number that takes the place of the glob in
        # DATA_PATHS) to a human-readable name.
        TASK_NAMES = {
            1: "First-last",
            2: "Strictly non-local",
            3: "Unbounded",
            4: "2-local",
            5: "Majority",
        }
    elif experiment in NL_NAMES:
        EXPERIMENT_HEADER = "*/nl_" + NL_NAMES[experiment]
        TASK_NAMES = {
            1 : NL_NAMES[experiment]
        }
    else:
        assert(0), f"Unknown experiment {experiment}"

    DATA_PATHS = {
        "SIP-ISL": f"{EXPERIMENT_HEADER}_*local_isl_isl_*",
        "SIP-FST": f"{EXPERIMENT_HEADER}_*SIP_None*",
        "SIP-TSL": f"{EXPERIMENT_HEADER}_*local_tsl_tsl_canon*",
        "ByT5": f"{EXPERIMENT_HEADER}_*t5_None*",
    }

def metric2label(metric: Literal["acc", "edit_dist", "inform"]) -> str:
    """Maps from an internal metric name to a more plot-friendly label."""
    return {
        "acc": "Accuracy",
        "inform": "Informedness",
        "edit_dist": "Edit distance",
    }[metric]


def title2nicer(label: str) -> str:
    """Maps from the label of an automatically generated plot title to a nicer one.

    Keeps `label` as-is if no nicer title is found."""
    return {
        "ae->aA": "a → A / e _",
        "a.e->a.A": "a → A / e [ ] _",
        "a..e->a..A": "a → A / e [ ] [ ] _",
        "a.*e->a.*A": "a → A / e [ ] * _",
        "english2": "English",
        "german-syll": "German",
        "polish": "Polish",
        "turkish-childes": "Turkish",
        "finnish" : "Finnish",
        "da->Da" : "Reg. (d → D / _ a)",
        "ad->aD" : "Prog. (d → D / a _)",

    }.get(label, label)


def loadScores(root: str) -> pd.DataFrame:
    """Loads scores_new.tsv from each of `DATA_PATHS` under `root`.

    Columns are added to the data for the model and task. The model and task
    names are inferred from the directory names.

    Returns a data frame with all rows from each TSV.
    """
    frames = []
    for model, pattern in DATA_PATHS.items():
        for data_home in glob.glob(os.path.join(root, pattern)):
            if "nl_" in EXPERIMENT_HEADER:
                task_name = EXPERIMENT_HEADER.replace("*/nl_", "")
            else:
                try:
                    task_index = int(
                        re.search(rf"{EXPERIMENT_HEADER}_(\d)", data_home).group(1))
                    task_name = TASK_NAMES[task_index]
                except Exception:
                    raise ValueError(
                        f"Unable to infer task name from {pattern}. "
                        "Try editing TASK_NAMES in this script."
                    )

            try:
                cipher = re.search(r"ciph_((monoalphabetic_)?\d)", data_home).group(1)
                #print("cipher found is", cipher)
            except Exception:
                cipher = 0

            try:
                local_data = pd.read_csv(
                    os.path.join(data_home, "scores_new.tsv"),
                    sep="\t",
                )
            except FileNotFoundError:
                local_data = pd.read_csv(
                    os.path.join(data_home, "scores.tsv"),
                    sep="\t",
                )

            local_data.insert(0, "model", model)
            local_data.insert(0, "task", task_name)
            local_data.insert(0, "cipher", cipher)
            frames.append(local_data)
    return pd.concat(frames)

def plotSingle(
    data: pd.DataFrame,
    task: Literal["ae->aA", "a.e->a.A", "a.?e->a.?A", "a.*e->a.*A"],
    model: Literal["SIP-ISL", "SIP-FST", "ByT5"],
    metric: Literal["acc", "edit_dist", "inform"],
    cipher: Literal["facet", "aggregate", "remove"],
):
    """Plots a single metric from `data` with bootstrapped confidence region.

    X axis values are `data["num_train"]`, and Y axis values are
    `data[metric]`.

    Returns the plot.
    """
    assert(cipher != "facet"), "Can't facet a single plot."

    selectors = (data.task == task) & (data.model == model)
    g = sns.relplot(
        data=data[selectors], kind="line", estimator="median",
        errorbar=("pi", 50), x="num_train", y=metric, err_style="bars")
    g.set(xlabel="Fine-tuning size", ylabel=metric2label(metric))
    g.facet_axis(0, 0).set_title(title2nicer(task))
    return g


def plotModelByTaskGrid(
    data: pd.DataFrame,
    models: List[Literal["SIP-ISL", "SIP-FST", "ByT5"]],
    tasks: List[Literal["ae->aA", "a.e->a.A", "a.?e->a.?A", "a.*e->a.*A"]],
    metric: Literal["acc", "edit_dist", "inform"],
    cipher: Literal["facet", "aggregate", "remove"],
):
    """Plots a grid of `models` (rows) by `tasks` (columns) from `data`.

    Returns the grid.
    """
    assert(cipher != "facet"), "Can't map cipher to a grid property because we're using both."
    selectors = (
        functools.reduce(
            lambda a, b: a | b, map(lambda t: data["task"] == t, tasks))
        & functools.reduce(
            lambda a, b: a | b, map(lambda m: data["model"] == m, models))
        # Silently filter out rows with NaNs.
        & (data[metric] == data[metric])
        & (data["num_train"] == data["num_train"])
    )
    selected = data[selectors]
    g = sns.FacetGrid(selected, col="task", row="model", margin_titles=True)
    g.map_dataframe(
        sns.lineplot, data=selected, estimator="median", errorbar=("pi", 50),
        x="num_train", y=metric, err_style="bars")
    g.set(xlabel="Fine-tuning size", ylabel=metric2label(metric))
    for i in range(len(tasks)):
        g.facet_axis(0, i).set_title(title2nicer(tasks[i]))
    return g


def plotModelByTaskOverlay(
    data: pd.DataFrame,
    models: List[Literal["SIP-ISL", "SIP-FST", "ByT5"]],
    tasks: List[Literal["ae->aA", "a.e->a.A", "a.?e->a.?A", "a.*e->a.*A"]],
    metric: Literal["acc", "edit_dist", "inform"],
    cipher: Literal["facet", "aggregate", "remove"],
    ticks: Literal["synthetic", "nl"],
    transpose: bool=False
):
    """Plots trend lines for multiple `models` on various `tasks` from `data`.

    Returns the plot.
    """
    selectors = (
        functools.reduce(
            lambda a, b: a | b, map(lambda t: data["task"] == t, tasks))
        & functools.reduce(
            lambda a, b: a | b, map(lambda m: data["model"] == m, models))
        # Silently filter out rows with NaNs.
        & (data[metric] == data[metric])
        & (data["num_train"] == data["num_train"])
    )
    selected = data[selectors]

    col = "task"
    hue = "model"
    col_order = tasks #in the order they appear
    hue_order = models
    
    if transpose:
        col, col_order, hue, hue_order = (hue, hue_order, col, col_order)

        #ugh, legend issues
        hue_order = [title2nicer(ii) for ii in hue_order]
        task2 = selected["task"].map(title2nicer)
        selected = selected.assign(task=task2)
        
    if cipher == "facet":
        ciphers = sorted(data["cipher"].unique(), key=str)
        g = sns.FacetGrid(selected, col=col, col_order=col_order, row="cipher", row_order=ciphers)
    else:
        g = sns.FacetGrid(selected, col=col, col_order=col_order)

    g.map_dataframe(
        sns.lineplot, data=selected, estimator="median", errorbar=("pi", 50),
        x="num_train", y=metric, err_style="band", hue=hue, hue_order=hue_order)

    g.add_legend()
        
    g.set(xlabel="Fine-tuning size", ylabel=metric2label(metric))

    for i in range(len(col_order)):
        if not transpose:
            g.facet_axis(0, i).set_title(title2nicer(tasks[i]))
        else:
            g.set_titles(template="{col_name}")

    for i in range(len(col_order)):
        if cipher == "facet":
            for j in range(1, len(ciphers)):
                g.facet_axis(j, i).set_title(ciphers[j].replace("ma_", "Cipher "))


    #TODOS: set axis orders
    #TODOS: allow transposed overlay plot
        
    #limit y axis
    if ticks == "synthetic":
        g.set(ylim=[.6, 1])
        g.set(xlim=[0, 32])
        g.set(xticks=range(0, 33, 8))
    else:
        g.set(xticks=sorted(data["num_train"].unique()))
        g.tick_params(axis="x", rotation=-90)
        g.tight_layout()
        
    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="plot_amp")
    parser.add_argument("data_root", help="Location of log file subdirs")
    parser.add_argument(
        "--metric",
        choices=["acc", "edit_dist", "inform"],
        default="inform",
        help="Metric to plot",
    )
    parser.add_argument(
        "-e",
        "--experiment",
        choices=EXPERIMENTS,
        required=True,
        help="Experiment to select tasks from",
    )
    parser.add_argument(
        "-t",
        "--task",
        action="extend",
        type=str,
        nargs="+",
        required=True,
        help="Task(s) to plot",
    )
    parser.add_argument(
        "-m",
        "--model",
        action="extend",
        type=str,
        nargs="+",
        required=True,
        help="Model(s) to plot",
    )
    parser.add_argument(
        "-c",
        "--cipher",
        choices=["facet", "aggregate", "remove"],
        required=True,
        help="How to handle cipher experiments"
    )
    parser.add_argument(
        "--multi",
        choices=["grid", "overlay", "overlay-model"],
        default="grid",
        help=("Whether to split multiple relationships (as when more than one "
              "task or model is specified) into a grid or overlay them on one "
              "plot"),
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Supress legend and figure captions")
    parser.add_argument(
        "-o", "--out",
        required=True,
        help="File to write plot to",
    )
    args = parser.parse_args()

    set_experiment_paths(args.experiment)
    data = loadScores(args.data_root)
    if args.cipher == "remove":
        data = data.loc[data["cipher"] == 0,]
    else:
        ma_cipher = ((data.cipher == 0) | data.cipher.map(lambda xx: type(xx) == str and "monoalphabetic" in xx))
        data = data[ma_cipher]
        data.cipher = data.cipher.map(lambda xx: xx if xx == 0 else xx.replace("monoalphabetic", "ma"))

    if "all" in args.task:
        tasks = sorted(TASK_NAMES.keys())
    else:
        #allow user to specify tasks in order
        tasks = [int(xx) for xx in args.task]
    tasks = [TASK_NAMES.get(xx) for xx in tasks]
    if "all" in args.model:
        models = sorted(DATA_PATHS.keys())
    else:
        models = sorted(set(args.model))

    sns.set_style("whitegrid")

    if args.experiment in NL_NAMES:
        ticks = "nl"
    else:
        ticks = "synthetic"
    
    if len(tasks) > 1 or len(models) > 1:
        if args.multi == "grid":
            plotModelByTaskGrid(
                data, models=models, tasks=tasks, metric=args.metric, cipher=args.cipher,
            ).figure.savefig(args.out)
        elif args.multi == "overlay":
            plotModelByTaskOverlay(
                data, models=models, tasks=tasks, metric=args.metric, cipher=args.cipher, ticks=ticks,
            ).figure.savefig(args.out)
        elif args.multi == "overlay-model":
            plotModelByTaskOverlay(
                data, models=models, tasks=tasks, metric=args.metric, cipher=args.cipher, ticks=ticks, transpose=True
            ).figure.savefig(args.out)
        else:
            plotSingle(
                data, task=tasks[0], model=models[0], metric=args.metric, cipher=args.cipher,
            ).figure.savefig(args.out)
