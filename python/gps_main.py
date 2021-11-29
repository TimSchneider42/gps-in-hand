#!/usr/bin/env python3

""" This file defines the main object that runs experiments. """
import csv
import shlex
import signal
from collections import defaultdict
from ctypes import c_bool
from multiprocessing.pool import Pool
from shutil import ignore_patterns
from threading import Lock
from time import sleep

import matplotlib as mpl

from gps.config import DebugCostFunction
from gps.sample import SampleList

mpl.use("QT5Agg")
from multiprocessing import Value

import sys

from gps import Config, Experiment
from gps.algorithm import Algorithm, IterationData, ConditionData, Trajectory, TrajectoryCost
from gps.gui import OnlineGUI, OfflineGUI, DetailLevel
from gps.utility.iteration_database import IterationDatabase, EntryState

import matplotlib.pyplot as plt

import glob
from typing import Optional, List, NamedTuple, Dict

import shutil

import logging
import importlib.util
import os
import argparse
import numpy as np


def run_training(experiment_name: str, output_root: str, run_name: str, iteration_database: IterationDatabase,
                 show_gui: bool, resume_training_itr: Optional[int], quit_on_end: bool, config_args: str):
    run_directory = os.path.join(output_root, experiment_name, run_name)

    experiment = __create_experiment(iteration_database, experiment_name, config_args, run_directory)

    with open(os.path.join(os.path.join(output_root, experiment_name), ".latest_run"), "w") as f:
        f.write(run_name)

    if show_gui:
        gui = OnlineGUI(experiment, run_name, resume_training_itr, wait_on_start=False, quit_on_end=quit_on_end)
        gui.show()
    else:
        def abort(signal, frame):
            experiment.abort()

        experiment.setup(run_name=run_name, itr_load=resume_training_itr)
        for s in [signal.SIGABRT, signal.SIGQUIT, signal.SIGHUP, signal.SIGINT]:
            signal.signal(s, abort)
        experiment.run()


def __itr_num(iteration_file: str):
    return int(os.path.split(iteration_file)[-1].split(".")[0].split("_")[-1])


def __get_latest_run(experiment_directory: str) -> str:
    latest_run_file = os.path.join(experiment_directory, ".latest_run")
    if not os.path.isfile(latest_run_file):
        logging.error("No run present to load for experiment \"{}\"".format(os.path.split(experiment_directory)[-1]))
        exit(1)
    else:
        with open(latest_run_file) as f:
            run_name = f.read()
        return run_name


def __get_highest_iteration(iteration_files: List[str]):
    return __itr_num(list(sorted(iteration_files))[-1])


def __get_iteration_files(experiment_directory: str, run_name: str) -> List[str]:
    run_directory = os.path.join(experiment_directory, run_name)
    path = os.path.join(run_directory, "data", "iteration_[0-9][0-9][0-9].pkl")
    return glob.glob(path)


def __load_config(config_directory: str, args: str) -> Config:
    config_filename = os.path.join(config_directory, "config.py")

    spec = importlib.util.spec_from_file_location("config", config_filename)
    experiment_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(experiment_config)
    return experiment_config.create_config(shlex.split(args))


def __create_experiment(iteration_database: IterationDatabase, experiment_name: str, config_args: str,
                        run_directory: str) -> Experiment:
    # Load config
    config = __load_config(os.path.join(run_directory, "config"), config_args)
    return Experiment(iteration_database=iteration_database,
                      name=experiment_name,
                      algorithm=config.algorithm,
                      num_iterations=config.iterations)


def __load_config_args(output_root: str, experiment_name: str, run_name: str):
    with open(os.path.join(output_root, experiment_name, run_name, "args")) as f:
        return f.read()


def __configure_logger(silent: bool, run_directory: Optional[str] = None, run_name: Optional[str] = None):
    logging.basicConfig(level=logging.DEBUG)
    logging.root.handlers.clear()

    sh_formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s ")

    sh_stderr = logging.StreamHandler(sys.stderr)
    sh_stderr.setLevel(logging.ERROR)
    sh_stderr.setFormatter(sh_formatter)
    logging.root.addHandler(sh_stderr)

    sh_stdout = logging.StreamHandler(sys.stdout)
    sh_stdout.setLevel(logging.INFO if silent else logging.DEBUG)
    sh_stdout.setFormatter(sh_formatter)
    logging.root.addHandler(sh_stdout)

    if run_name is not None and run_directory is not None:
        fh = logging.FileHandler(os.path.join(run_directory, "log_{}.txt".format(run_name)))
        fh_formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s ")
        fh.setFormatter(fh_formatter)
        fh.setLevel(logging.DEBUG)
        logging.root.addHandler(fh)


def resume(args):
    output_root = os.path.abspath(args.output_root)
    experiment_directory = os.path.join(output_root, args.config)
    run_name = args.run
    if run_name is None:
        run_name = __get_latest_run(experiment_directory)
    run_directory = os.path.join(experiment_directory, run_name)
    if not os.path.exists(run_directory):
        logging.error("Run \"{}\" for experiment \"{}\" does not exist.".format(run_name, args.config))
        exit(1)
    iteration = args.iteration

    iteration_database = IterationDatabase(os.path.join(run_directory, "data"),
                                           keep_newest_only=not args.keep_all_controllers,
                                           default_entry_state=EntryState.NOT_LOADED)
    iteration_database.load()

    if len(iteration_database.full_iteration_numbers) == 0:
        logging.error("Found no full iterations for run \"{}\" on experiment \"{}\".".format(run_name, args.config))
        exit(1)

    if iteration is not None:
        if iteration not in [it.iteration_no for it in iteration_database.entries]:
            logging.error("Iteration {} for run \"{}\" on experiment \"{}\" does not exist.".format(
                iteration, run_name, args.config))
            exit(1)
        elif not iteration_database.entries[iteration].full_data_available:
            logging.error("Only the cache of iteration {} for run \"{}\" on experiment \"{}\" is present.".format(
                iteration, run_name, args.config))
            exit(1)

    highest_iteration = iteration_database.full_iteration_numbers[-1]
    if iteration is None:
        iteration = highest_iteration
    elif iteration < highest_iteration:
        # Figure out location to copy run to
        i = 0
        while os.path.exists(os.path.join(experiment_directory, "{}_{:03d}".format(run_name, i))):
            i += 1
        new_run_name = "{}_{:03d}".format(run_name, i)
        # Create copy
        shutil.copytree(os.path.join(experiment_directory, run_name),
                        os.path.join(experiment_directory, new_run_name))
        # Delete all unnecessary iterations
        path = os.path.join(os.path.join(experiment_directory, new_run_name), "data",
                            "iteration_[0-9][0-9][0-9].pkl")
        iteration_files = glob.glob(path)
        for f in iteration_files:
            if __itr_num(f) > iteration:
                os.remove(f)

        run_name = new_run_name
        run_directory = os.path.join(experiment_directory, run_name)

        iteration_database = IterationDatabase(os.path.join(run_directory, "data"),
                                               default_entry_state=EntryState.NOT_LOADED)
        iteration_database.load()

    __configure_logger(args.silent, run_directory, run_name)
    logging.info("Resuming run \"{}\" of experiment\"{}\" at iteration {}".format(
        run_name, args.config, iteration))

    run_training(args.config, output_root, run_name, iteration_database, not args.no_gui, iteration, args.quit,
                 config_args=__load_config_args(output_root, args.config, run_name))


def create(args):
    output_root = os.path.abspath(args.output_root)
    config_root = os.path.abspath(args.config_root)
    experiment_directory = os.path.join(output_root, args.config)
    latest_id_file = os.path.join(experiment_directory, ".latest_id")
    run_id = 0
    if os.path.exists(latest_id_file):
        with open(latest_id_file) as f:
            run_id = int(f.read()) + 1

    run_name = args.run
    if run_name is None:
        run_name = "{id}"
    # Probe if setting the id makes a difference (checking if the user used {id} in the name)
    if run_name.format(id="0") != run_name.format(id="1"):
        # Look for the next free name
        while os.path.exists(
                os.path.join(experiment_directory, run_name.format(id="{:04d}".format(run_id)))):
            run_id += 1
    run_name_final = run_name.format(id="{:04d}".format(run_id))
    os.makedirs(os.path.dirname(latest_id_file), exist_ok=True)
    with open(latest_id_file, "w") as f:
        f.write(str(run_id))
    run_directory = os.path.join(experiment_directory, run_name_final)
    if os.path.exists(run_directory):
        logging.error("Run \"{}\" for experiment \"{}\" already exists.".format(run_name_final, args.config))
        exit(1)
    os.makedirs(run_directory)
    with open(os.path.join(run_directory, "args"), "w") as f:
        f.write(args.args)

    __configure_logger(args.silent, run_directory, run_name_final)

    # Create new run
    config_path = os.path.join(config_root, args.config)
    if not os.path.exists(config_path):
        logging.error(
            "Experiment \"{0}\" does not exist in directory \"{1}\".".format(args.config, config_root))
        exit(1)
    config_filename = os.path.join(config_path, "config.py")
    if not os.path.exists(config_filename):
        logging.error("Experiment \"{0}\" has no config.py.".format(args.config))
        exit(1)

    # Copy config to output directory
    shutil.copytree(os.path.join(os.path.abspath(config_root), args.config),
                    os.path.join(run_directory, "config"))
    logging.info("Starting run \"{}\" of experiment \"{}\"".format(run_name_final, args.config))

    iteration_database = IterationDatabase(os.path.join(run_directory, "data"),
                                           keep_newest_only=not args.keep_all_controllers,
                                           default_entry_state=EntryState.NOT_LOADED)
    iteration_database.initialize_new()

    run_training(args.config, output_root, run_name_final, iteration_database, not args.no_gui, None, args.quit,
                 args.args)


def test(args):
    __configure_logger(args.silent)
    output_root = os.path.abspath(args.output_root)
    experiment_directory = os.path.join(output_root, args.config)

    repetitions = args.repetitions
    run_name = __get_latest_run(experiment_directory) if args.run is None else args.run
    run_directory = os.path.join(experiment_directory, run_name)
    if not os.path.exists(run_directory):
        logging.error("Run \"{}\" for experiment \"{}\" does not exist.".format(run_name, args.config))
        exit(1)

    iteration_database = IterationDatabase(os.path.join(run_directory, "data"),
                                           default_entry_state=EntryState.NOT_LOADED)
    iteration_database.load()

    if len(iteration_database.full_iteration_numbers) == 0:
        logging.error(
            "No iteration found for run \"{}\" on experiment \"{}\"".format(run_name, args.config))
        exit(1)

    if not args.override_args:
        config_args = args.args + " " + __load_config_args(output_root, args.config, run_name)
    else:
        config_args = args.args

    iteration = iteration_database.full_iteration_numbers[-1] if args.iteration is None else args.iteration

    entry = iteration_database.entries[iteration]
    entry.state = EntryState.LOADED
    iteration_data = entry.iteration_data

    # TODO: remove
    config = __load_config(os.path.join(run_directory, "config"), config_args)
    experiment = __create_experiment(iteration_database, args.config, config_args, run_directory)

    agent = experiment.agent
    aborted = Value(c_bool, False)
    abort_lock = Lock()

    try:
        agent.initialize()
        for cmd in args.debug_cmds:
            agent.debug_command(cmd)

        def abort(signal, frame, aborted=aborted):
            if not aborted.value:
                with abort_lock:
                    aborted.value = True
                    agent.abort()

        for s in [signal.SIGABRT, signal.SIGQUIT, signal.SIGHUP, signal.SIGINT]:
            signal.signal(s, abort)

        if not args.interactive:
            default_controllers = {
                c: c if iteration_data.algorithm_data.policy is None else None for c in
                experiment.algorithm.training_conditions + experiment.algorithm.test_conditions
            }
            default_controllers.update({c: None for c in experiment.algorithm.policy_conditions})

            if args.conditions is not None:
                conditions = [(int(c[0]) if len(c) > 1 else default_controllers[int(c[-1])], int(c[-1])) for c in
                              [s.split(":") for s in args.conditions]]
            else:
                conditions = [(p, c) for c, p in default_controllers.items()]

            logging.info(
                """Running tests:
                 Experiment:     {exp}
                 Run:            {run}
                 Iteration:      {it}
                 Conditions:     {cond}
                 Repetitions:    {rep}""".format(exp=experiment.name, run=run_name, it=iteration,
                                                 cond=", ".join([f"{c[0]}:{c[1]}" for c in conditions]),
                                                 rep=repetitions))

            for r in range(repetitions):
                for p, c in conditions:
                    logging.info(f"Running repetition {r} on condition {c} using "
                                 f"{f'distribution {p}' if p is not None else 'policy'}...")
                    with abort_lock:
                        if aborted.value:
                            break
                        policy = iteration_data.algorithm_data.policy
                        if policy is None or p is not None:
                            policy = iteration_data.cond_data[c if p is None else p].algorithm_data.traj_distr
                        agent.setup(policy, c)
                    sample = agent.run()
                    if aborted.value:
                        break
                    l = experiment.algorithm.conditions[c].cost_function.eval(sample)[0]
                    logging.info(f"Repetition {r} on condition {c} using "
                                 f"{f'distribution {p}' if p is not None else 'policy'} completed. Cost: {np.sum(l)}.")
                if aborted.value:
                    break
        else:
            while True:
                try:
                    pc = input("Enter condition code to run: ")
                    pcs = pc.split(":")
                    p = int(pcs[0]) if len(pcs) > 1 else None
                    c = int(pcs[-1])
                    logging.info(f"Running test on condition {c} using "
                                 f"{f'distribution {p}' if p is not None else 'policy'}...")
                    with abort_lock:
                        if aborted.value:
                            break
                        policy = iteration_data.algorithm_data.policy
                        if policy is None or p is not None:
                            policy = iteration_data.cond_data[c if p is None else p].algorithm_data.traj_distr
                        agent.setup(policy, c)
                    sample = agent.run()
                    if aborted.value:
                        break
                    l = experiment.algorithm.conditions[c].cost_function.eval(sample)[0]
                    logging.info(f"Test on condition {c} using "
                                 f"{f'distribution {p}' if p is not None else 'policy'} completed. Cost: {np.sum(l)}.")
                    for dcf in config.debug_cost_functions[c]:
                        v = dcf.cost.eval(sample)[0]
                        print(f"{dcf.desc}: sum: {np.sum(v)} final: {v[-1]}")
                except:
                    print("Invalid code. Valid syntax is [CONTROLLER:]CONDITION")
                    continue

        if aborted.value:
            logging.info("Testing aborted.")
    finally:
        agent.terminate()


def test_policy(args):
    __configure_logger(args.silent)
    output_root = os.path.abspath(args.output_root)
    experiment_directory = os.path.join(output_root, args.config)

    input_run_name = __get_latest_run(experiment_directory) if args.run is None else args.run
    input_run_directory = os.path.join(experiment_directory, input_run_name)

    if not os.path.exists(input_run_directory):
        logging.error("Run \"{}\" for experiment \"{}\" does not exist.".format(input_run_name, args.config))
        exit(1)

    input_iteration_database = IterationDatabase(os.path.join(input_run_directory, "data"),
                                                 default_entry_state=EntryState.NOT_LOADED)
    input_iteration_database.load()

    if len(input_iteration_database.full_iteration_numbers) == 0:
        logging.error(
            "No iteration found for run \"{}\" on experiment \"{}\"".format(input_run_name, args.config))
        exit(1)

    if not args.override_args:
        config_args = args.args + " " + __load_config_args(output_root, args.config, input_run_name)
    else:
        config_args = args.args

    experiment = __create_experiment(input_iteration_database, args.config, config_args, input_run_directory)
    agent = experiment.agent

    conditions = experiment.algorithm.policy_conditions if args.conditions is None else args.conditions

    i = 0
    while os.path.exists(os.path.join(experiment_directory, f"{input_run_name}_{i:02d}")):
        i += 1
    output_run_name = f"{input_run_name}_{i:02d}"
    output_run_directory = os.path.join(experiment_directory, output_run_name)
    os.mkdir(output_run_directory)
    shutil.copytree(os.path.join(input_run_directory, "config"), os.path.join(output_run_directory, "config"))
    with open(os.path.join(output_run_directory, "args"), "w") as f:
        f.write(config_args)

    output_iteration_database = IterationDatabase(os.path.join(output_run_directory, "data"),
                                                  default_entry_state=EntryState.NOT_LOADED)
    output_iteration_database.initialize_new()

    aborted = Value(c_bool, False)
    abort_lock = Lock()

    def abort(signal, frame, aborted=aborted):
        if not aborted.value:
            with abort_lock:
                aborted.value = True
                agent.abort()

    for s in [signal.SIGABRT, signal.SIGQUIT, signal.SIGHUP, signal.SIGINT]:
        signal.signal(s, abort)

    cost_course = np.zeros(len(input_iteration_database.full_iteration_numbers))
    try:
        agent.initialize()
        for i, iteration in enumerate(input_iteration_database.full_iteration_numbers):
            entry = input_iteration_database.entries[iteration]
            entry.state = EntryState.LOADED
            iteration_data = entry.iteration_data
            policy = iteration_data.algorithm_data.policy
            condition_data = {}
            for c in conditions:
                logging.info(f"Iteration {iteration} - condition {c}...")
                with abort_lock:
                    if aborted.value:
                        break
                    agent.setup(policy, c)
                sample = agent.run()
                if aborted.value:
                    break
                cost = experiment.algorithm.conditions[c].cost_function.eval(sample)
                cost = TrajectoryCost(*cost, None, None, None)

                mean_cost = np.sum(cost.l)
                cost_course[i] += mean_cost

                traj = Trajectory(None, policy, [cost], mean_cost, SampleList([sample]), None, None, None, None, None,
                                  None, None)

                if c in iteration_data.cond_data:
                    cd = iteration_data.cond_data[c]
                    condition_data[c] = ConditionData(cd.algorithm_data, cd.control_noise_rng_state,
                                                      cd.training_trajectory, cd.test_trajectory, traj)
                else:
                    condition_data[c] = ConditionData(None, None, None, None, traj)
                logging.info("Completed.")
            data = IterationData(iteration_data.iteration_no, iteration_data.algorithm_data, condition_data)
            output_iteration_database.store(data)
            entry.state = EntryState.NOT_LOADED
    finally:
        agent.terminate()
    cost_course /= len(conditions)
    plt.plot(range(len(cost_course)), cost_course)
    plt.show()


def test_setup(args):
    __configure_logger(args.silent)
    config_root = os.path.abspath(args.config_root)
    config_path = os.path.join(config_root, args.config)

    config_args = args.args

    config = __load_config(config_path, config_args)

    agent = config.algorithm.agent

    conditions = list(range(agent.condition_count) if args.conditions is None else args.conditions)

    try:
        agent.initialize()

        if args.cmd is not None:
            for cmd in args.cmd:
                agent.debug_command(cmd)

        while True:
            for c in conditions:
                logging.info(f"Condition {c}")
                agent._reset(c)
                if args.interactive:
                    logging.info("Press any key to continue or enter command...")
                    cmd = None
                    while cmd != "":
                        cmd = input()
                        if cmd != "":
                            try:
                                agent.debug_command(cmd)
                            except:
                                print("Invalid command!")
                else:
                    sleep(3.0)
                    while len(conditions) == 1:
                        sleep(3.0)

    finally:
        agent.terminate()


def print_summary(algorithm: Algorithm, iteration_data: List[IterationData]):
    """
    Setup iteration data column titles: iteration, average cost, and for
    each condition the mean cost over samples, step size, linear Guassian
    controller entropies, and initial/final KL divergences for BADMM.
    """
    data_description = [d[1:] for d in algorithm.display_data_description if not d[0]]
    condition_data_description = [d[1:] for d in algorithm.display_data_description if d[0]]
    condition_data_length = {}

    itr_data_fields = " ".join(["{n: >{c}}".format(c=c, n=n) for i, n, f, c in data_description])
    condition_titles = " " * len(itr_data_fields)

    cond_fields = " ".join(["{n: >{c}}".format(c=c, n=n) for i, n, f, c in condition_data_description])

    conditions = sorted(
        set(algorithm.training_conditions).union(algorithm.test_conditions).union(algorithm.policy_conditions))

    for cond in conditions:
        cond_title = "condition {}".format(cond)

        length = max(len(cond_fields), len(cond_title))
        condition_data_length[cond] = length

        condition_titles += " | " + cond_title.rjust(length)
        itr_data_fields += " | " + cond_fields.rjust(length)
    output_text_header = condition_titles + os.linesep + itr_data_fields
    output_text_lines = []
    for itr_data in iteration_data:
        disp = itr_data.display_data
        data = " ".join([f(disp[i]).rjust(c) for i, n, f, c in data_description])
        for cond in conditions:
            cond_disp = defaultdict(lambda : np.nan, itr_data.cond_data[cond].display_data)
            cond_data = " ".join([f(cond_disp[i]).rjust(c) for i, n, f, c in condition_data_description])
            data += " | " + cond_data.rjust(condition_data_length[cond])
        output_text_lines.append(data)
    print(output_text_header)
    for l in output_text_lines:
        print(l)


def show(args):
    output_root = os.path.abspath(args.output_root)
    experiment_directory = os.path.join(output_root, args.config)

    run_name = args.run
    if run_name is None:
        run_name = __get_latest_run(experiment_directory)
    run_directory = os.path.join(experiment_directory, run_name)
    if not os.path.exists(run_directory):
        logging.error("Run \"{}\" for experiment \"{}\" does not exist.".format(run_name, args.config))
        exit(1)

    iteration_database = IterationDatabase(os.path.join(run_directory, "data"),
                                           default_entry_state=EntryState.REDUCED_LOADED)
    iteration_database.load()

    experiment = __create_experiment(iteration_database, args.config,
                                     __load_config_args(output_root, args.config, run_name), run_directory)

    print_summary(experiment.algorithm, [e.iteration_data for e in iteration_database.entries.values()])
    detail_level = DetailLevel.NO_DETAILS
    if args.detailed_cost:
        detail_level = DetailLevel.COST
    if args.detailed_policies:
        detail_level = DetailLevel.COST_AND_POLICY
    gui = OfflineGUI(iteration_database, experiment, run_name, detail_level=detail_level)
    gui.show()


_TrajectoryCost = NamedTuple("_TrajectoryCost", (("cost", np.ndarray), ("sum", float), ("avg", float)))


def _create_traj_cost(cost: np.ndarray):
    return _TrajectoryCost(cost, np.sum(cost), np.average(cost))


def _compute_dcf_traj_cost(iteration_no: int, iteration_database: IterationDatabase,
                           dcfs: Dict[int, List[DebugCostFunction]]):
    print(f"Iteration {iteration_no} started...")
    entry = iteration_database.entries[iteration_no]
    entry.state = EntryState.REDUCED_LOADED
    itr = entry.iteration_data
    output = {
        c: {
            dcf.desc: {
                n: _create_traj_cost(np.average([dcf.cost.eval(s)[0] for s in t.samples], axis=0))
                for t, n in [(itr.cond_data[c].training_trajectory, "training"),
                             (itr.cond_data[c].test_trajectory, "test"),
                             (itr.cond_data[c].policy_trajectory, "policy")]
                if t is not None
            } for dcf in dcfs_c
        } for c, dcfs_c in dcfs.items()
    }
    entry.state = EntryState.NOT_LOADED
    print(f"Iteration {iteration_no} done.")
    return output


def analyze_cost(args):
    output_root = os.path.abspath(args.output_root)
    experiment_directory = os.path.join(output_root, args.config)

    run_name = args.run
    if run_name is None:
        run_name = __get_latest_run(experiment_directory)
    run_directory = os.path.join(experiment_directory, run_name)
    if not os.path.exists(run_directory):
        logging.error("Run \"{}\" for experiment \"{}\" does not exist.".format(run_name, args.config))
        exit(1)

    print("Loading database...")
    iteration_database = IterationDatabase(os.path.join(run_directory, "data"),
                                           default_entry_state=EntryState.NOT_LOADED)
    iteration_database.load()
    print("Done loading database.")

    iteration_no = iteration_database.full_iteration_numbers[-1] if args.iteration is None else args.iteration

    iteration_numbers = [e.iteration_no for e in iteration_database.entries.values()]

    if not args.override_args:
        config_args = args.args + " " + __load_config_args(output_root, args.config, run_name)
    else:
        config_args = args.args

    config = __load_config(os.path.join(run_directory, "config"), config_args)

    conditions = list(range(config.algorithm.agent.condition_count) if args.conditions is None else args.conditions)

    dcfs = {
        c: [dcf for dcf in [DebugCostFunction(config.algorithm.conditions[c].cost_function, "actual_cost", True)] +
            config.debug_cost_functions[c] if args.dcf is None or dcf.desc in args.dcf]
        for c in conditions
    }

    # Compute all cost functions for all trajectories for all conditions in every iteration
    print("Computing cost...")
    pool = Pool(processes=7)
    cost_data_res = {
        i: pool.apply_async(_compute_dcf_traj_cost, (i, iteration_database, dcfs))
        for i in iteration_numbers
    }
    cost_data_get = {i: res.get() for i, res in cost_data_res.items()}

    print("Reorganizing cost...")

    cost_data = {
        c: {
            dcf: {
                itr: cost_data_get[itr][c][dcf.desc] for itr in iteration_numbers
            } for dcf in dcfs[c]
        } for c in conditions
    }
    print("Done computing cost.")

    print("Portions of absolute total cost:")
    for c in conditions:
        trajs = []
        if c in config.algorithm.training_conditions:
            trajs.append("training")
        if c in config.algorithm.test_conditions:
            trajs.append("test")
        if c in config.algorithm.policy_conditions:
            trajs.append("policy")

        # Create cost function composition overview for this iteration
        for n in trajs:
            print(f"  Condition {c} - {n}:")
            cost_sums = []
            total_sum = 0
            for dcf, itr_cost in cost_data[c].items():
                cost = itr_cost[iteration_no][n].cost
                cost_sum = abs(sum(cost))
                cost_sums.append((dcf.desc, cost_sum))
                if not dcf.exclude_from_total_sum:
                    total_sum += cost_sum
            for cn, s in cost_sums:
                print(f"    {cn}: {s/total_sum * 100.0}%")

        # Plot detailed cost for this iteration
        for n in trajs:
            plt.figure()
            plt.title(f"Condition {c} - {n}")
            for dcf, itr_cost in cost_data[c].items():
                cost = itr_cost[iteration_no][n].cost
                plt.plot(range(0, len(cost)), cost, label=dcf.desc)
            plt.figlegend()

        # Write CSV for this iteration
        with open(os.path.join(run_directory, f"c{c}_itr{iteration_no:03d}.csv"), "w") as f:
            writer = csv.writer(f, delimiter=',')
            complete_data = [(dcf, n, cost.cost) for dcf, itr_cost in cost_data[c].items() for n, cost in
                             itr_cost[iteration_no].items()]
            # write header
            writer.writerow(["ts"] + [f"{dcf.desc}_{n}" for dcf, n, d in complete_data])
            for i in range(config.algorithm.agent.time_steps):
                writer.writerow([i] + [d[i] for dcf, n, d in complete_data])

        # Plot course of cost for all iterations
        for n in trajs:
            # plot
            plt.figure()
            plt.title(f"Over iterations: condition {c} - {n}")
            for dcf, itr_cost in cost_data[c].items():
                dnp = np.array(
                    [(i, itr_cost[i][n].avg) for i in iteration_numbers])
                plt.plot(dnp[:, 0], dnp[:, 1], label=dcf.desc)
            plt.figlegend()

        # Write CSV for all iterations
        with open(os.path.join(run_directory, f"c{c}.csv"), "w") as f:
            writer = csv.writer(f, delimiter=',')
            num_itr = len(iteration_numbers)
            complete_data = [(dcf, n, np.array([itr_cost[i][n].avg for i in iteration_numbers]))
                             for dcf, itr_cost in cost_data[c].items() for n in trajs]
            # write header
            writer.writerow(["itr"] + [f"{dcf.desc}_{n}" for dcf, n, d in complete_data])
            for i in range(num_itr):
                writer.writerow([i] + [d[i] for dcf, n, d in complete_data])

    dcf_names = set(["actual_cost"] + [dcf.desc for c in conditions for dcf in config.debug_cost_functions[c]])
    entries = defaultdict(lambda: [])
    for d in dcf_names:
        for c in conditions:
            itr_data = [i for dcf, i in cost_data[c].items() if dcf.desc == d]
            if len(itr_data) > 0:
                if c in config.algorithm.training_conditions:
                    entries[d + "_training"].append(("training", itr_data[0]))
                if c in config.algorithm.test_conditions:
                    entries[d + "_test"].append(("test", itr_data[0]))
                if c in config.algorithm.policy_conditions:
                    entries[d + "_policy"].append(("policy", itr_data[0]))
                if c in config.algorithm.policy_conditions and c not in config.algorithm.training_conditions:
                    entries[d + "_policy_test"].append(("policy", itr_data[0]))
                if c in config.algorithm.policy_conditions and c in config.algorithm.training_conditions:
                    entries[d + "_policy_train"].append(("policy", itr_data[0]))

    entry_list = sorted(entries.items(), key=lambda x: x[0])
    for n, fs in [("avg", lambda x: x.avg), ("sum", lambda x: x.sum), ("final", lambda x: x.cost[-1])]:
        with open(os.path.join(run_directory, f"{n}.csv"), "w") as f:
            writer = csv.writer(f, delimiter=',')
            # write header
            writer.writerow(["itr"] + [e for d in entry_list for e in [d[0] + "_avg", d[0] + "_stddev"]])
            for itr in iteration_numbers:
                row = [e for desc, itr_data_arr in entry_list for e in [
                    np.average([fs(itr_data[itr][n]) for n, itr_data in itr_data_arr]),
                    np.std([fs(itr_data[itr][n]) for n, itr_data in itr_data_arr])]]
                writer.writerow([itr] + row)
    if not args.quit:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Guided Policy Search algorithm.")
    parser.add_argument("config", type=str, metavar="CONFIG",
                        help="Name of the configuration to run.")
    parser.add_argument("output_root", type=str, metavar="OUT_DIR",
                        help="Root directory for the output files.")
    parser.add_argument("-s", "--silent", action="store_true", help="Silent debug print outs.")

    subparsers = parser.add_subparsers()

    parser_create = subparsers.add_parser("create", help="Create a new run.")
    parser_create.add_argument("-r", "--run", type=str,
                               help="Name of the new run. This name is unique and must not exist yet. To use a unique "
                                    "identifier, use the format specifier {id}. (default: \"{id}\")")
    parser_create.add_argument("--no-gui", action="store_true", help="Do not show GUI.")
    parser_create.add_argument("-q", "--quit", action="store_true", help="Quit GUI automatically when finished.")
    parser_create.add_argument("--args", type=str, default="", help="Arguments passed to the configuration file.")
    parser_create.add_argument("-k", "--keep-all-controllers", action="store_true",
                               help="Keep all controllers instead of the newest ones only. This option will take "
                                    "significantly more memory.")
    parser_create.add_argument("config_root", metavar="CONFIG_DIR", type=str,
                               help="Root directory of the experiment configurations.")
    parser_create.set_defaults(func=create)

    parser_resume = subparsers.add_parser("resume", help="Resume a previous run.")
    parser_resume.add_argument("-r", "--run", type=str, help="Name of the run to resume (default: latest run).")
    parser_resume.add_argument("-i", "--iteration", type=int,
                               help="Resume training from this iteration (default: last iteration). To avoid a loss of "
                                    "data the algorithm will be run on a copy <run_name>_XXX of the data if this option"
                                    " is specified and does not point to the highest iteration.")
    parser_resume.add_argument("--no-gui", action="store_true", help="Do not show GUI.")
    parser_resume.add_argument("-q", "--quit", action="store_true", help="Quit GUI automatically when finished.")
    parser_resume.add_argument("-k", "--keep-all-controllers", action="store_true",
                               help="Keep all controllers instead of the newest ones only. This option will take "
                                    "significantly more memory.")

    parser_resume.set_defaults(func=resume)

    parser_show = subparsers.add_parser("show", help="Visualize an existing run.")
    parser_show.add_argument("-r", "--run", type=str, help="Name of the run to visualize (default: latest run).")
    parser_show.add_argument("-c", "--conditions", type=int, nargs="+", help="Conditions to visualize (default: all).")
    mutex_group_show = parser_show.add_mutually_exclusive_group()
    mutex_group_show.add_argument("--detailed-cost", action="store_true", help="Show cost details for each condition.")
    mutex_group_show.add_argument("--detailed-policies", action="store_true",
                                  help="Unstable feature: Show cost and controller details for each condition.")
    parser_show.set_defaults(func=show)

    parser_analyze_cost = subparsers.add_parser("analyze_cost", help="Analyze cost of one iteration in detail.")
    parser_analyze_cost.add_argument("-r", "--run", type=str,
                                     help="Name of the run to visualize (default: latest run).")
    parser_analyze_cost.add_argument("-c", "--conditions", type=int, nargs="+",
                                     help="Conditions to visualize (default: all).")
    parser_analyze_cost.add_argument("--iteration", type=int,
                                     help="Iteration to take test controller from (default: highest).")
    parser_analyze_cost.add_argument("-d", "--dcf", type=str, nargs="+", help="Debug cost functions to consider.")
    parser_analyze_cost.add_argument("--args", type=str, default="",
                                     help="Additional arguments for the experiment configuration.")
    parser_analyze_cost.add_argument("--override-args", action="store_true",
                                     help="Override args instead of appending them.")
    parser_analyze_cost.add_argument("-q", "--quit", action="store_true",
                                     help="Quit when finished with reports and do not show graphs.")
    parser_analyze_cost.set_defaults(func=analyze_cost)

    parser_test = subparsers.add_parser("test", help="Test the controller of a given iteration.")
    parser_test.add_argument("-r", "--run", type=str, help="Name of the run to test (default: latest run).")
    parser_test.add_argument("--iteration", type=int,
                             help="Iteration to take test controller from (default: highest).")
    parser_test.add_argument("-i", "--interactive", action="store_true",
                             help="Interactive mode, most arguments will be ignored")
    parser_test.add_argument("-c", "--conditions", type=str, nargs="+", default=None,
                             help="""Conditions to test and which controller to use.
Syntax: [CONTROLLER:]CONDITION [CONTROLLER:]CONDITION ...
Example:    "3:4 4:1 2" to test condition 4 with controller 3, 
            condition 1 with controller 4 and condition 2 with the policy      
(default: test all conditions with their respective controllers).""")
    parser_test.add_argument("-n", "--repetitions", type=int, default=10,
                             help="Number of repetitions of the test (default: 10).")
    parser_test.add_argument("--args", type=str, default="",
                             help="Additional arguments for the experiment configuration.")
    parser_test.add_argument("--override-args", action="store_true", help="Override args instead of appending them.")
    mutex_group_test = parser_test.add_mutually_exclusive_group()  # TODO: implement
    mutex_group_test.add_argument("-p", "--pause-time", type=float, default=1.0,
                                  help="Seconds to pause between two tests (default: 1.0)")
    mutex_group_test.add_argument("-w", "--wait-for-input", action="store_true",
                                  help="Wait for keystroke between two tests.")
    parser_test.add_argument("-d", "--debug-cmds", type=str, nargs="+", default=[],
                             help="Debug commands for the agent.")
    parser_test.set_defaults(func=test)

    parser_test = subparsers.add_parser("test_policy", help="Test the policy for all iterations.")
    parser_test.add_argument("-r", "--run", type=str, help="Name of the run to test (default: latest run).")
    parser_test.add_argument("-c", "--conditions", type=int, nargs="+", default=None,
                             help="Conditions to test policy on.")
    parser_test.add_argument("--args", type=str, default="",
                             help="Additional arguments for the experiment configuration.")
    parser_test.add_argument("--override-args", action="store_true", help="Override args instead of appending them.")
    parser_test.set_defaults(func=test_policy)

    parser_test_setup = subparsers.add_parser("test_setup", help="Test an agent setup.")
    parser_test_setup.add_argument("config_root", metavar="CONFIG_DIR", type=str,
                                   help="Root directory of the experiment configurations.")
    parser_test_setup.add_argument("-c", "--conditions", type=int, nargs="+", default=None,
                                   help="Conditions to test (default: all).")
    parser_test_setup.add_argument("--cmd", type=str, nargs="+", default=None,
                                   help="Commands to send to the agent upon start.")
    parser_test_setup.add_argument("--args", type=str, default="", help="Arguments for the experiment configuration.")
    parser_test_setup.add_argument("-i", "--interactive", action="store_true",
                                   help="Enable interactive mode.")

    parser_test_setup.set_defaults(func=test_setup)

    args = parser.parse_args()

    args.func(args)

    logging.info("Terminated.")
