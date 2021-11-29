#!/usr/bin/env python3

import argparse
import json
import os
import math

import signal
import subprocess
from math import isnan
from time import sleep

from typing import List, Tuple, NamedTuple

from threading import Lock

import pickle
import numpy as np
from numpy import random
from subprocess import Popen, DEVNULL
import matplotlib as mpl

mpl.use("QT5Agg")
import matplotlib.pyplot as plt

from setuptools import glob

from gps.algorithm import IterationData

Result = NamedTuple("Result", (
    ("run_index", int), ("return_value", int), ("iterations", int), ("mean_test_cost", float),
    ("test_cost", np.ndarray),
    ("mean_training_cost", float), ("training_cost", np.ndarray),
    ("seed", int)))


def __get_last_meta_run_no(root_dir):
    last_meta_run_filename = os.path.join(root_dir, ".last_meta_run")
    try:
        with open(last_meta_run_filename) as f:
            return int(f.read())
    except Exception:
        return 0


def __create_new_meta_run(root_dir):
    os.makedirs(root_dir, exist_ok=True)
    root_dir = os.path.abspath(root_dir)
    last_meta_run_filename = os.path.join(root_dir, ".last_meta_run")

    last_meta_run = __get_last_meta_run_no(root_dir)

    while os.path.exists(__get_meta_run_path(root_dir, last_meta_run)):
        last_meta_run += 1

    with open(last_meta_run_filename, "w") as f:
        f.write(str(last_meta_run))

    return __get_meta_run_path(root_dir, last_meta_run)


def __get_meta_run_path(root_dir, run_id):
    return os.path.abspath(os.path.join(root_dir, f"meta_{run_id:04d}"))


def __retrieve_test_result(run_dir):
    itr_data_path = os.path.join(run_dir, "data")
    itr_files = sorted(glob.glob(os.path.join(itr_data_path, "iteration_*.pkl")))
    if len(itr_files) > 0:
        with open(itr_files[-1], "rb") as f:
            itr_data: IterationData = pickle.load(f)
        test_cost = np.array([d.test_trajectory.mean_cost for c, d in itr_data.cond_data.items() if
                              d.test_trajectory is not None])
        if len(test_cost) == 0:
            mean_test_cost = np.nan
        else:
            mean_test_cost = np.mean(test_cost)
        training_cost = np.array([d.training_trajectory.mean_cost for c, d in itr_data.cond_data.items() if
                                  d.training_trajectory is not None])
        if len(training_cost) == 0:
            mean_training_cost = np.nan
        else:
            mean_training_cost = np.mean(training_cost)
        iterations = itr_data.iteration_no
    else:
        test_cost = training_cost = mean_training_cost = mean_test_cost = np.nan
        iterations = 0
    meta_info_path = os.path.join(run_dir, "meta.json")
    if os.path.exists(meta_info_path):
        with open(meta_info_path) as f:
            meta_dict = json.load(f)
    else:
        # Extract seed from args
        with open(os.path.join(run_dir, "args")) as f:
            seed = int(f.read().split(" ")[2])
        meta_dict = {"return_value": 0, "hp_seed": seed}
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta_dict, f, indent=2)
    return Result(int(os.path.split(run_dir)[-1]), meta_dict["return_value"], iterations, mean_test_cost, test_cost,
                  mean_training_cost, training_cost, meta_dict["hp_seed"])


def print_results(results):
    result_list = sorted(results, key=lambda r: r.mean_test_cost if not isnan(r.mean_test_cost) else float("inf"))
    term = "\033[0m"
    bold = "\033[1m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"

    tst_c = [r.mean_test_cost for r in result_list if not isnan(r.mean_test_cost)]
    best_tst_cost = min(tst_c) if len(tst_c) > 0 else np.nan
    tr_c = [r.mean_training_cost for r in result_list if not isnan(r.mean_training_cost)]
    best_tr_cost = min(tr_c) if len(tr_c) > 0 else np.nan

    print("  i  | itr | test cost  | train cost | ret | seed")
    print("-----+-----+------------+------------+-----+-----")
    for r in result_list:
        color = green if r.return_value == 0 else red
        tst_color = bold if r.mean_test_cost == best_tst_cost else ""
        tr_color = bold if r.mean_training_cost == best_tr_cost else ""
        print(f"{color}{r.run_index: >4} | {r.iterations: >3} | {tst_color}{r.mean_test_cost:10.3f}{term}{color} | "
              f"{tr_color}{r.mean_training_cost:10.3f}{term}{color} | {r.return_value: ^3} | {r.seed}{term}")


aborting = False


def run(args):
    global aborting
    output_dir = __create_new_meta_run(args.output_dir)

    abort_lock = Lock()

    def abort(signum, frame):
        global aborting
        with abort_lock:
            aborting = True
            for i, seed, p in active_processes:
                p.send_signal(signum)

    for s in [signal.SIGABRT, signal.SIGQUIT, signal.SIGHUP, signal.SIGINT]:
        signal.signal(s, abort)

    runs = args.runs
    threads = args.num_threads
    current_run = 0
    active_processes: List[Tuple[int, int, Popen]] = []

    results: List[Result] = []

    def check_termination(active_processes):
        terminated_thread_indices = []
        for i, (index, seed, t) in enumerate(active_processes):
            ret_val = t.poll()
            if ret_val is not None:
                terminated_thread_indices.append(i)
                run_dir = os.path.join(output_dir, CONFIG_NAME, f"{index:04d}")
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "meta.json"), "w") as f:
                    json.dump({"hp_seed": seed, "return_value": ret_val}, f, indent=2)
                results.append(__retrieve_test_result(run_dir))
                print(f"Run {index} terminated with exit code {ret_val}")
                if not aborting:
                    print("Current results:")
                    print_results(results)
        for i in reversed(terminated_thread_indices):
            active_processes[i:i + 1] = []

    while not len(active_processes) == 0 or current_run < runs and not aborting:
        check_termination(active_processes)
        while len(active_processes) == threads:
            sleep(0.5)
            check_termination(active_processes)
        with abort_lock:
            if not aborting and current_run < runs:
                seed = random.randint(0, 2 ** 32 - 1)
                print(f"Starting run {current_run} using hp-seed {seed}")
                p = subprocess.Popen(
                    ["./gps_main.py", CONFIG_NAME, output_dir, "create", "../config", "-q", "--args",
                     f"--rand-hp --hp-seed {seed} {args.args}", "-r", f"{current_run:04d}"],
                    stdout=DEVNULL)
                active_processes.append((current_run, seed, p))
                current_run += 1

    print()
    print("Results:")
    print_results(results)


def __to_num(value):
    try:
        return int(value)
    except ValueError:
        return float(value)


def __retrieve_hyperparameters(run_dir):
    hp_filename = os.path.join(run_dir, "config", "hyperparameters.json")
    if os.path.exists(hp_filename):
        with open(hp_filename) as f:
            return json.load(f)
    else:
        # Parse logs
        with open(os.path.join(run_dir, f"log_{os.path.split(run_dir)[-1]}.txt")) as f:
            log = f.readlines()
        hyperparam_lines = [l for l in log if "config.py: " in l][1:]
        hyperparam_lines = [l.split("config.py: ")[-1] for l in hyperparam_lines]
        initial_state_start = None
        for i, l in enumerate(hyperparam_lines):
            if l.startswith("Initial states:"):
                initial_state_start = i
                break
        hyperparam_lines = hyperparam_lines[:initial_state_start]
        hyperparam_lines_split = [l.split(": ") for l in hyperparam_lines]
        hp_dict = {n: __to_num(v) for n, v in hyperparam_lines_split}
        with open(hp_filename, "w") as f:
            json.dump(hp_dict, f, indent=2)
        return hp_dict


def __plot_param(name, values, test_cost, training_cost):
    if not all(v == values[0] for v in values):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(name)
        if isinstance(values[0], int) or 0.0 in values:
            ax.scatter(values, test_cost, label="test")
            ax.scatter(values, training_cost, label="training")
            ax.set_xlabel("Value")
        else:
            ax.scatter([math.log10(v) for v in values], test_cost, label="test")
            ax.scatter([math.log10(v) for v in values], training_cost, label="training")
            ax.set_xlabel("Value (log10)")
        ax.legend()
        ax.set_ylabel("Cost")
    else:
        print(f"Skipping plot of {name} as all parameter values are equal.")


def analyze(args):
    if args.id is None:
        run_no = __get_last_meta_run_no(args.output_dir)
    else:
        run_no = args.id
    analyze_dir = os.path.join(__get_meta_run_path(args.output_dir, run_no), CONFIG_NAME)
    runs = [d for d in [os.path.join(analyze_dir, d) for d in os.listdir(analyze_dir)] if os.path.isdir(d)]
    results = [(__retrieve_test_result(d), __retrieve_hyperparameters(d)) for d in runs]
    print_results(list(zip(*results))[0])
    # Remove outliers
    results = [r for r in results if r[0].mean_test_cost < 500]
    if args.parameters is not None:
        plot_params = args.parameters
    else:
        plot_params = results[0][1].keys()

    for p in plot_params:
        values = [d[1][p] for d in results]
        test_cost = np.array([r[0].mean_test_cost for r in results]).reshape((-1,))
        training_cost = np.array([r[0].mean_training_cost for r in results]).reshape((-1,))
        __plot_param(p, values, test_cost, training_cost)
    plt.show()


parser = argparse.ArgumentParser(
    description="Meta runner for gps_main.py using allegro_cylinder_pybullet_meta configuration.")

parser.add_argument("output_dir", type=str, help="Output directory.")

subparsers = parser.add_subparsers()

parser_run = subparsers.add_parser("run", help="Start a new meta-run")
parser_run.add_argument("-r", "--runs", type=int, default=float("inf"), help="Number of runs (default: infinite).")
parser_run.add_argument("-t", "--num-threads", type=int, default=1, help="Number of parallel runs (default: 1).")
parser_run.add_argument("--args", type=str, default="", help="Arguments for each run.")
parser_run.set_defaults(func=run)

parser_analyze = subparsers.add_parser("analyze", help="Analyze a previous run.")
parser_analyze.add_argument("-i", "--id", type=int, default=None, help="Id of the meta run to analyze.")
parser_analyze.add_argument("-p", "--parameters", type=str, nargs="+", help="Parameter names to plot.")
parser_analyze.set_defaults(func=analyze)

args = parser.parse_args()

CONFIG_NAME = "allegro_pybullet"

args.func(args)
