import os
from pathlib import Path
import hydra
from loguru import logger
import pandas as pd

from report.plots import MakePlots


@hydra.main(version_base=None, config_path="config_files", config_name="cdeath")
def collect_report(config):
    # Add scores to results table
    path_results = Path(os.path.join(config.meta.out_dir, f"results_table.xlsx"))
    res_table = pd.read_excel(path_results)
    scores = ["sts", "loes", "es2"]
    score_names = ["STS-PROM", "Log ES", "ES 2"]
    for i in range(len(scores)):
        score = scores[i]
        score_filepath = path_results.parent.parent / (path_results.parent.name + f"_{score}") / path_results.name
        score_table = pd.read_excel(score_filepath)
        row = score_table[score_table["train_strategy"] == "refit"].copy()
        row["Selector"] = score_names[i]
        res_table = pd.concat([res_table, row], axis=0)
    res_table.to_excel(os.path.join(config.meta.out_dir, f"results_table_scores.xlsx"), index=False)
    # Make plots
    logger.info(f"Making plots for {config.meta.in_file}...")
    MakePlots(config)()


if __name__ == "__main__":
    collect_report()
