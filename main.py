import os
import sys
import hydra
from omegaconf import OmegaConf
import enlighten
import numpy as np
from loguru import logger

from preprocessing.preprocessing import Preprocessing
from survival.train_evaluate import Survival


@hydra.main(version_base=None, config_path="config_files", config_name="cdeath")
def main(config):
    if config.meta.out_dir is None:
        config.meta.out_dir = os.path.splitext(config.meta.in_file)[0]
    os.makedirs(config.meta.out_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.meta.out_dir, "config.yaml"))
    logfile_path = os.path.join(config.meta.out_dir, "logs.log")
    file = open(logfile_path, 'a')
    sys.stderr = file
    sys.stdout = file
    logger.add(logfile_path)
    progress_manager = enlighten.get_manager()
    seed = config.meta.seed
    preprocessing = Preprocessing(config)
    pipeline = Survival(config, progress_manager)
    logger.info(f'Running seed {seed}')
    np.random.seed(seed)
    data_x_train, data_x_test, data_y_train, data_y_test, comp_label_train, comp_label_test = preprocessing()
    _ = pipeline(
        seed,
        data_x_train,
        data_y_train,
        data_x_test,
        data_y_test,
        comp_label_train,
        comp_label_test
    )


if __name__ == "__main__":
    main()
