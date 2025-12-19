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


# Maps from model name to a glob pattern for experiment results. Extend this as
# more models are added.
DATA_PATHS = {
    "SIP-ISL": "harmony_*_local_isl_isl_markov",
    "SIP-FST": "harmony_*_SIP_None",
    "t5": "harmony_*_t5_None",
}


# Maps from task index (an arbitrary number that takes the place of the glob in
# DATA_PATHS) to a human-readable name.
TASK_NAMES = {
    1: "ae->aA",
    2: "a.e->a.A",
    3: "a.?e->a.?A",
    4: "a.*e->a.*A",
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
        "a.?e->a.?A": "a → A / e [ ] [ ] _",
        "a.*e->a.*A": "a → A / e [ ] * _",
        "english2": "English",
        "german-syll": "German",
        "polish": "Polish",
        "turkish-childes": "Turkish",
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
            try:
                task_index = int(
                    re.search(r"harmony_(\d)", data_home).group(1))
                task_name = TASK_NAMES[task_index]
            except Exception:
                raise ValueError(
                    f"Unable to infer task name from {pattern}. "
                    "Try editing TASK_NAMES in this script."
                )
            local_data = pd.read_csv(
                os.path.join(data_home, "scores_new.tsv"),
                sep="\t",
            )
            local_data.insert(0, "model", model)
            local_data.insert(0, "task", task_name)
            frames.append(local_data)
    return pd.concat(frames)


def plotSingle(
    data: pd.DataFrame,
    task: Literal["ae->aA", "a.e->a.A", "a.?e->a.?A", "a.*e->a.*A"],
    model: Literal["SIP-ISL", "SIP-FST", "t5"],
    metric: Literal["acc", "edit_dist", "inform"],
):
    """Plots a single metric from `data` with bootstrapped confidence region.

    X axis values are `data["num_train"]`, and Y axis values are
    `data[metric]`.

    Returns the plot.
    """
    selectors = (data.task == task) & (data.model == model)
    g = sns.relplot(
        data=data[selectors], kind="line", estimator="median",
        errorbar=("pi", 50), x="num_train", y=metric, err_style="bars")
    g.set(xlabel="Fine-tuning size", ylabel=metric2label(metric))
    g.facet_axis(0, 0).set_title(title2nicer(task))
    return g


def plotModelByTaskGrid(
    data: pd.DataFrame,
    models: List[Literal["SIP-ISL", "SIP-FST", "t5"]],
    tasks: List[Literal["ae->aA", "a.e->a.A", "a.?e->a.?A", "a.*e->a.*A"]],
    metric: Literal["acc", "edit_dist", "inform"],
):
    """Plots a grid of `models` (rows) by `tasks` (columns) from `data`.

    Returns the grid.
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
    models: List[Literal["SIP-ISL", "SIP-FST", "t5"]],
    tasks: List[Literal["ae->aA", "a.e->a.A", "a.?e->a.?A", "a.*e->a.*A"]],
    metric: Literal["acc", "edit_dist", "inform"],
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
    g = sns.FacetGrid(selected, col="task")
    g.map_dataframe(
        sns.lineplot, data=selected, estimator="median", errorbar=("pi", 50),
        x="num_train", y=metric, err_style="band", hue="model")
    g.add_legend()
    g.set(xlabel="Fine-tuning size", ylabel=metric2label(metric))
    for i in range(len(tasks)):
        g.facet_axis(0, i).set_title(title2nicer(tasks[i]))
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
        "-t",
        "--task",
        choices=list(TASK_NAMES.values()) + ["all"],
        action="extend",
        type=str,
        nargs="+",
        required=True,
        help="Task(s) to plot",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=list(DATA_PATHS.keys()) + ["all"],
        action="extend",
        type=str,
        nargs="+",
        required=True,
        help="Model(s) to plot",
    )
    parser.add_argument(
        "--multi",
        choices=["grid", "overlay"],
        default="grid",
        help=("Whether to split multiple relationships (as when more than one "
              "task or model is specified) into a grid or overlay them on one "
              "plot"),
    )
    parser.add_argument(
        "-o", "--out",
        required=True,
        help="File to write plot to",
    )
    args = parser.parse_args()

    data = loadScores(args.data_root)

    if "all" in args.task:
        tasks = sorted(TASK_NAMES.values())
    else:
        tasks = sorted(set(args.task))
    if "all" in args.model:
        models = sorted(DATA_PATHS.keys())
    else:
        models = sorted(set(args.model))

    sns.set_style("whitegrid")

    if len(tasks) > 1 or len(models) > 1:
        match args.multi:
            case "grid":
                plotModelByTaskGrid(
                    data, models=models, tasks=tasks, metric=args.metric
                ).figure.savefig(args.out)
            case "overlay":
                plotModelByTaskOverlay(
                    data, models=models, tasks=tasks, metric=args.metric
                ).figure.savefig(args.out)
    else:
        plotSingle(
            data, task=tasks[0], model=models[0], metric=args.metric
        ).figure.savefig(args.out)
