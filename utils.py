from pathlib import Path
import json
from pathlib import Path
from typing import Tuple

import pandas as pd


def unflatten_metadata(metadata) -> Tuple[dict, dict]:
    res = {}
    if "config_general" in metadata:
        res["model_dtype"] = metadata["config_general"].get("model_dtype")
        res["model_size"] = metadata["config_general"].get("model_size")
    if "summary_general" in metadata:
        res["total_evaluation_time_secondes"] = metadata['summary_general']['total_evaluation_time_secondes']
    return res


def load_leaderboard_records(path: Path):
    records = []
    for result_file in path.rglob("*.json"):
        with open(result_file, "r") as f:
            metadata = json.load(f)
        metadata["filename"] = result_file  # no loss of information as the name file encodes some information
        if "config_general" in metadata and "results" in metadata:
            records.append(metadata)
    return records


def convert_dataframe(results_records: list):
    rows = []
    for result_record in results_records:
        # The schema was changed at some point ... see discussion https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/discussions/367
        runtime = result_record.get('config_general', {}).get('total_evaluation_time_secondes')
        if not runtime:
            runtime = result_record.get('summary_general', {}).get('total_evaluation_time_secondes')
        metadata = {
            # "organization": organization,
            "model_name": result_record["config_general"]["model_name"],
            "model_dtype": result_record["config_general"].get("model_dtype"),
            "model_size": result_record["config_general"].get("model_size"),
            "total_evaluation_time_secondes": runtime,
            "date": Path(result_record["filename"]).stem.replace("results_", "").split(".")[0]
        }
        """
        * arc, hellaswag, mmlu/hendrycksTest -> acc_norm
        * truthful_qa -> mc2
        * drop -> em
        * gms8k, winogrande -> acc
        """
        for benchmark, metrics in result_record["results"].items():
            if benchmark != "all":
                row = dict(**metadata)
                for metric, value in metrics.items():
                    benchmark_family = benchmark.replace("harness|", "") if "hendrycksTest" not in benchmark else "hendrycksTest|5"
                    # TODO switch case with new python
                    if "drop" in benchmark_family:
                        expected_benchmark_metric = "f1"
                    elif "gsm8k" in benchmark_family or "winogrande" in benchmark_family:
                        expected_benchmark_metric = "acc"
                    elif "truthfulqa" in benchmark_family:
                        expected_benchmark_metric = "mc2"
                    else:
                        expected_benchmark_metric = "acc_norm"
                    if metric == expected_benchmark_metric:
                        row["metric"] = value
                        # harness|arc:challenge|25 -> arc:challenge|25
                        row["benchmark"] = benchmark.replace("harness|", "")

                        # ugly special case hendrycksTest-world_religions -> hendrycksTest
                        if "hendrycksTest" in benchmark:
                            row["benchmark_family"] = "hendrycksTest|5"
                        else:
                            row["benchmark_family"] = row["benchmark"]
                        rows.append(row)
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"].apply(lambda x: x.split("T")[0]), format="%Y-%m-%d", errors="coerce")
    # df = df["date"]
    return df


def load_dataframe_results(path):
    return convert_dataframe(load_leaderboard_records(path))


if __name__ == '__main__':
    # TODO load only last model

    # run before !git clone https://huggingface.co/datasets/open-llm-leaderboard/results
    records = load_leaderboard_records(Path("results/"))
    # print(records)

    df = convert_dataframe(records)
    df[df.model_name.str.contains("Qwen/Qwen-72B")]
    # print(df.benchmark_family.unique())
    # print(df.to_string())

