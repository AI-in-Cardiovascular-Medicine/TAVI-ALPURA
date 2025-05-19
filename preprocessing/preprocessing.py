import os
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from missforest import MissForest
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None


class Preprocessing:
    def __init__(self, config) -> None:
        self.in_file = config.meta.in_file
        self.test_file = config.meta.test_file
        self.event_column = config.meta.events
        self.time_column = config.meta.times
        self.comp_event_column = config.meta.competing_events
        self.columns_to_drop = config.preprocessing.columns_to_drop
        self.corr_threshold = config.preprocessing.corr_threshold
        self.replace_zero_time_with = config.preprocessing.replace_zero_time_with
        self.out_dir = config.meta.out_dir

    def __call__(self):
        # Check if the imputed data file already exists
        data_out_file = os.path.join(self.out_dir, "data_imputed.pkl")
        if os.path.exists(data_out_file):
            logger.info(f"Found existing imputed data file at {data_out_file}, loading data.")
            with open(data_out_file, 'rb') as f:
                data_dict = pickle.load(f)
            # Assign the loaded data directly
            self.data_x_train = data_dict["data_x_train"]
            self.data_x_test = data_dict["data_x_test"]
            self.data_y_train = data_dict["data_y_train"]
            self.data_y_test = data_dict["data_y_test"]
            self.comp_event_label_train = data_dict.get("comp_event_label_train", None)
            self.comp_event_label_test = data_dict.get("comp_event_label_test", None)
        else:
            self.load_data()  # load data
            self.load_test()  # load test data
            self.impute_data()  # impute data
            self.remove_highly_correlated_features()
            data_dict = {
                "data_x_train": self.data_x_train,
                "data_x_test": self.data_x_test,
                "data_y_train": self.data_y_train,
                "data_y_test": self.data_y_test,
                "comp_event_label_train": self.comp_event_label_train,
                "comp_event_label_test": self.comp_event_label_test
            }
            data_out_file = os.path.join(self.out_dir, "data_imputed.pkl")
            os.makedirs(self.out_dir, exist_ok=True)
            with open(data_out_file, 'wb') as f:
                pickle.dump(data_dict, f)
            logger.info(f'Saved data split to {data_out_file}')
        self.data_y_train = self.to_structured_array(self.data_y_train)  # scikit-survival requires structured array
        self.data_y_test = self.to_structured_array(self.data_y_test)
        return (self.data_x_train, self.data_x_test, self.data_y_train, self.data_y_test, self.comp_event_label_train,
                self.comp_event_label_test)

    def load_data(self):
        try:
            data = pd.read_excel(self.in_file)
            data = data.apply(pd.to_numeric, errors='coerce')  # replace non-numeric entries with NaN
            data = data.dropna(how='all', axis=1)  # drop columns with all NaN
            data.columns = [col.replace(" ", "_") for col in data.columns]
            # Drop columns from predictors
            cols_to_drop = [self.time_column, self.event_column]
            if self.columns_to_drop is not None and len(self.columns_to_drop) > 0:
                cols_to_drop += self.columns_to_drop
                logger.info(f"Dropping features {self.columns_to_drop}.")
            self.data_x = data.drop(columns=cols_to_drop)
            if self.comp_event_column is not None:
                self.data_x = self.data_x.drop(columns=[self.comp_event_column])
                self.comp_event_label = data[self.comp_event_column]
            self.data_y = data[[self.event_column, self.time_column]]
            self.data_y[self.time_column] = self.data_y[self.time_column].replace(
                0, self.replace_zero_time_with
            )  # some models do not accept t <= 0 -> set to small value > 0
        except FileNotFoundError:
            logger.error(f'File {self.in_file} not found, check the path in the config.yaml file.')
            raise

    def load_test(self):
        """If external test data is provided, load it (no train-test split)."""
        self.data_x_train = self.data_x
        self.comp_event_label_train = self.comp_event_label
        self.data_y_train = self.data_y
        try:
            data = pd.read_excel(self.test_file)
            data = data.apply(pd.to_numeric, errors='coerce')  # replace non-numeric entries with NaN
            data = data.dropna(how='all', axis=1)  # drop columns with all NaN
            data.columns = [col.replace(" ", "_") for col in data.columns]
            # Drop columns from predictors
            cols_to_drop = [self.time_column, self.event_column]
            if self.columns_to_drop is not None and len(self.columns_to_drop) > 0:
                cols_to_drop += self.columns_to_drop
            self.data_x_test = data.drop(columns=cols_to_drop)
            if self.comp_event_column is not None:
                self.data_x_test = self.data_x_test.drop(columns=[self.comp_event_column])
                self.comp_event_label_test = data[self.comp_event_column]
            self.data_y_test = data[[self.event_column, self.time_column]]
            self.data_y_test[self.time_column] = self.data_y_test[self.time_column].replace(
                0, self.replace_zero_time_with
            )  # some models do not accept t <= 0 -> set to small value > 0
        except FileNotFoundError:
            logger.error(f'File {self.in_file} not found, check the path in the config.yaml file.')
            raise

    def impute_data(self):
        logger.info(f"Imputing data with MissForest")
        nunique = self.data_x_train.nunique()
        categorical = list(nunique[nunique < 10].index)
        if len(categorical) == 0:
            categorical = None
        self.imputer = MissForest(categorical=categorical)
        if self.data_x_train.isna().sum().sum() > 0:
            self.imputer.fit(self.data_x_train)
            imp_train = self.imputer.transform(self.data_x_train)
            self.data_x_train = pd.DataFrame(imp_train, index=self.data_x_train.index,
                                             columns=self.data_x_train.columns)
        if self.data_x_test.isna().sum().sum() > 0:
            imp_test = self.imputer.transform(self.data_x_test)
            self.data_x_test = pd.DataFrame(imp_test, index=self.data_x_test.index, columns=self.data_x_test.columns)

    def remove_highly_correlated_features(self):
        corr_matrix = self.data_x_train.corr()  # correlation matrix
        # compute importance based on c-index in cox univariate model
        c_index = {}
        for feature in self.data_x_train.columns:
            df_cox = pd.concat([self.data_x_train[feature], self.data_y_train], axis=1)
            cph = CoxPHFitter()
            cph.fit(df_cox, duration_col=self.time_column, event_col=self.event_column, formula=feature, fit_options={"step_size": 0.1})
            c_index[feature] = cph.concordance_index_
        importance = pd.Series(c_index).sort_values(ascending=False)
        # create upper triangle matrix and order by c-index. Then loop over features from the most important to the
        # least important and remove a feature if highly correlated to another one with higher c-index
        corr_matrix = corr_matrix.reindex(index=importance.index, columns=importance.index).abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > self.corr_threshold)]
        if len(to_drop) > 0:
            logger.info(f"Removing {len(to_drop)} highly correlated features: {to_drop}")
        self.data_x_train = self.data_x_train.drop(columns=to_drop)
        self.data_x_test = self.data_x_test.drop(columns=to_drop)

    def to_structured_array(self, df):
        return np.array(
            list(zip(df[self.event_column], df[self.time_column])),
            dtype=[(self.event_column, '?'), (self.time_column, '<f8')],
        )
