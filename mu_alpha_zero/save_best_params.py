import optuna
import json


def main():
    study = optuna.load_study(study_name="alpha_zero", storage="mysql://root:584792@localhost/alpha_zero")
    with open("best_params_2.json", "w") as file:
        json.dump(study.best_params, file)

    print(f"Best params saved to best_params.json. The params achieved a score of {study.best_value}.")


if __name__ == "__main__":
    main()