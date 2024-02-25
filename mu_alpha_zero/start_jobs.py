import argparse
import json
import os

import optuna

from mu_alpha_zero.AlphaZero.utils import az_optuna_parameter_search

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"


def main():
    parser_ = argparse.ArgumentParser(
        description="Start optuna optimization jobs"
    )
    # parser_.add_argument("n", help="Number of processes to start.")
    parser_.add_argument("t", help="Number of trials to run.")
    parser_.add_argument("-s", "--storage", help="The storage string to use.")
    parser_.add_argument("-n", "--study_name", help="The name of the optuna study to use.")
    parser_.add_argument("-i", "--init_net_path", help="The path to the initial network.")

    args = parser_.parse_args()
    # n = int(args.n)
    t = int(args.t)
    storage = args.storage
    if storage is None:
        storage = "mysql://root:584792@localhost/alpha_zero"
    study_name = args.study_name
    if study_name is None:
        study_name = "alpha_zero"

    init_net_path = args.init_net_path

    # print(f"Starting optuna optimization. Using parameters: \n"
    #       f"Number of parallel processes: {n}\n"
    #       f"Number of trials: {t}\n"
    #       f"Storage: {storage}\n"
    #       f"Study name: {study_name}\n"
    #       f"Starting...")
    az_optuna_parameter_search(n_trials=t, init_net_path=init_net_path, storage=storage, study_name=study_name)

    study = optuna.load_study(study_name=study_name, storage=storage)
    with open("best_params.json", "w") as file:
        json.dump(study.best_params, file)


if __name__ == "__main__":
    main()
