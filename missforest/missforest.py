import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


class MissForest:
    def __init__(
        self,
        initial_strategy_num="mean",
        initial_strategy_cat="constant",
        max_iter=150,
        n_imputations=10,
        keep_categorical=True,
        regressor_params=None,
        classifier_params=None,
    ):
        self.initial_strategy_num = initial_strategy_num
        self.initial_strategy_cat = initial_strategy_cat
        self.max_iter = max_iter
        self.n_imputations = n_imputations
        self.keep_categorical = keep_categorical
        self.regressor_params = regressor_params if regressor_params else {}
        self.classifier_params = classifier_params if classifier_params else {}

    def fit(self, X):
        # Store the column order
        self.column_order = X.columns.tolist()

        # Identify numerical and categorical columns
        self.num_cols = X.select_dtypes(include=np.number).columns
        self.cat_cols = X.select_dtypes(exclude=np.number).columns

        # Initialize IterativeImputers
        self.iter_imputers_num = [
            IterativeImputer(
                estimator=RandomForestRegressor(**self.regressor_params),
                initial_strategy=self.initial_strategy_num,
                max_iter=self.max_iter,
                missing_values=np.nan,
                random_state=i,
            )
            for i in range(self.n_imputations)
        ]
        self.iter_imputers_cat = [
            IterativeImputer(
                estimator=RandomForestClassifier(**self.classifier_params),
                initial_strategy=self.initial_strategy_cat,
                max_iter=self.max_iter,
                missing_values=-1,
                random_state=i,
            )
            for i in range(self.n_imputations)
        ]

        # Prepare the mapping for categorical columns
        self.cat_mapping = {}
        for col in self.cat_cols:
            X[col] = X[col].astype("category")
            self.cat_mapping[col] = dict(enumerate(X[col].cat.categories))

        # Fit IterativeImputers
        if self.num_cols.size > 0:
            for imputer in self.iter_imputers_num:
                imputer.fit(X[self.num_cols].dropna())
        if self.cat_cols.size > 0:
            for imputer in self.iter_imputers_cat:
                imputer.fit(X[self.cat_cols].dropna().apply(lambda x: x.cat.codes))

    def transform(self, X):
        X = X[self.column_order]
        X_copy = X.copy()

        # Impute numerical columns
        if self.num_cols.size > 0:
            X_copies_num = [
                imputer.transform(X[self.num_cols])
                for imputer in self.iter_imputers_num
            ]

        # Impute categorical columns
        if self.cat_cols.size > 0:
            X_copies_cat = [
                imputer.transform(X_copy[self.cat_cols].apply(lambda x: x.cat.codes))
                for imputer in self.iter_imputers_cat
            ]

        # Combine the imputed dataframes
        X_copies = [
            pd.concat(
                [
                    pd.DataFrame(X_copy_num, columns=self.num_cols),
                    pd.DataFrame(X_copy_cat, columns=self.cat_cols).astype("category"),
                ],
                axis=1,
            )
            for X_copy_num, X_copy_cat in zip(X_copies_num, X_copies_cat)
        ]

        if self.keep_categorical:
            # Convert the imputed values back to categories
            for X_copy in X_copies:
                for i, col in enumerate(self.cat_cols):
                    X_copy[col] = (
                        X_copy[col].map(self.cat_mapping[col]).astype("category")
                    )

        return X_copies

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
