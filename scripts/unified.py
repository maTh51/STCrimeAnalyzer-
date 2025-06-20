# unified_experiment.py

import os
import sys
import yaml
import warnings
import pandas as pd
import time
import psutil
import gpustat

from common.database import DatabaseConnection
from common.utils import pre_process, train_test_split
from common.grid import create_grid
from common.evaluation import EvaluationModel

from models.regressions.regressions import REGRESSIONS
from models.sthsl.class_auto_test import STHSLModel
from models.stkde.stkde_model import STKDEModel
from models.starima.starima import STARIMA

warnings.filterwarnings("ignore")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["EXPORT_CUDA"] = "<1>"

EXPERIMENTS = {
    "grid_size": {
        "values": [1000],
        "update_config": lambda config, val: config["evaluation"].update(
            {"grid_size": val}
        ),
        "column": "grid_size",
    },
    "data_volume": {
        "values": list(range(1, 11)),
        "modify_train": lambda train_points, val: train_points.sample(
            int(len(train_points) * val / 10)
        ),
        "column": "data_volume",
    },
    "temporal_granularity": {
        "values": ["8H", "1D", "7D"],
        "update_config": lambda config, val: config["evaluation"].update(
            {"temporal_granularity": val}
        ),
        "column": "temporal_granularity",
    },
    "temporal_distribution": {
        "values": [None],
        "update_config": lambda config, _: config,
        "column": None,
    },
    "temporal_resilience": {
        "values": [None],
        "update_config": lambda config, _: config,
        "column": None,
    },
    "temporal_window": {
        "values": list(
            zip(
                ["2024-06-01", "2024-04-01", "2024-01-01", "2023-07-01", "2022-07-01"],
                [30, 90, 180, 360, 720],
            )
        ),
        "update_config": lambda config, val: update_temporal_window(config, val),
        "column": "temporal_window",
    },
}


def update_temporal_window(config, val):
    start_date, days = val
    config["database"]["filters"]["start_date"] = start_date
    if "STKDEModel" in config["models"]:
        config["models"]["STKDEModel"]["days_past"] = days
        config["models"]["STKDEModel"]["slide_window"] = True
    return config


def run_experiment(config_path, save_path, experiment_name, steps=5):
    with open(config_path) as f:
        base_config = yaml.load(f, Loader=yaml.FullLoader)

    experiment = EXPERIMENTS[experiment_name]
    results = []

    for step in range(steps):
        tmp = pd.DataFrame()

        for val in experiment["values"]:
            with open(config_path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            if "update_config" in experiment:
                config = experiment["update_config"](config, val)

            print("Load Database...")
            df = DatabaseConnection().get_data_all(
                column=config["database"]["columns"],
                filter=config["database"]["filters"],
            )

            print("Pre process Database...")
            points = pre_process(
                df,
                config["database"]["filters"]["nome_municipio"],
                config["database"]["columns"],
            )
            train_points, test_points = train_test_split(
                points,
                config["evaluation"]["train_end_date"],
                config["evaluation"]["test_end_date"],
            )

            if experiment_name == "data_volume":
                train_points = experiment["modify_train"](train_points, val)

            print("Create Grid...")
            grid = create_grid(
                config["evaluation"]["grid_size"],
                config["database"]["filters"]["nome_municipio"],
            )

            time_start = time.time()
            gpu_stats_start = gpustat.GPUStatCollection.new_query()
            memory_usage_start = psutil.virtual_memory()

            print("Train Models...")
            models = []
            for m, params in config["models"].items():
                if "slide_window" in params:
                    del params["slide_window"]
                    model = globals()[m](
                        points=points,
                        grid=grid,
                        last_train_date=config["evaluation"]["train_end_date"],
                        temporal_granularity=config["evaluation"][
                            "temporal_granularity"
                        ],
                        **params,
                    )
                else:
                    model = globals()[m](
                        points=train_points,
                        grid=grid,
                        last_train_date=config["evaluation"]["train_end_date"],
                        temporal_granularity=config["evaluation"][
                            "temporal_granularity"
                        ],
                        **params,
                    )
                model.train()
                models.append(model)

            print("Evaluation Models...")
            eval_model = EvaluationModel(
                models,
                test_points,
                grid,
                config["evaluation"]["train_end_date"],
                config["evaluation"]["test_end_date"],
                config["evaluation"]["temporal_granularity"],
            )
            res = eval_model.simulate(
                hit_rate_percentage=config["evaluation"]["hit_rate_percentage"]
            )

            time_end = time.time()
            gpu_stats_end = gpustat.GPUStatCollection.new_query()
            memory_usage_end = psutil.virtual_memory()

            if experiment["column"]:
                res[experiment["column"]] = (
                    val if not isinstance(val, tuple) else val[0]
                )

            res["time"] = time_end - time_start
            res["memory_usage(%)"] = (
                memory_usage_end.percent - memory_usage_start.percent
            )
            res["memory_usage(GB)"] = res["memory_usage(%)"] * 0.62

            for gpu_start, gpu_end in zip(gpu_stats_start.gpus, gpu_stats_end.gpus):
                res[f"gpu{gpu_start.index}(%)"] = (
                    gpu_end.utilization - gpu_start.utilization
                )
                res[f"gpu{gpu_start.index}(GB)"] = (
                    res[f"gpu{gpu_start.index}(%)"] * 0.24
                )

            tmp = pd.concat([tmp, res])

        tmp = (
            tmp.groupby([experiment["column"], "model"]).mean().reset_index()
            if experiment["column"]
            else tmp.groupby(["model"]).mean().reset_index()
        )
        results.append(tmp)

    csv = pd.concat(results)
    model_suffix = "_".join(config["models"].keys())
    output_path = f"{save_path}_{model_suffix}.csv"
    csv.to_csv(output_path, index=False)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "./scripts/config-stkde.yaml"
    save_path = sys.argv[2] if len(sys.argv) > 2 else "./scripts/results/result"
    experiment = sys.argv[3] if len(sys.argv) > 3 else "grid_size"  # default experiment

    run_experiment(config_path, save_path, experiment)
