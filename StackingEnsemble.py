from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import numpy as np
import pandas as pd
from joblib import dump, load
import os
from scipy.stats import friedmanchisquare as friedman_test
import scikit_posthocs as sp
from sklearn.metrics import roc_auc_score
# import time, multiprocessing

model_name = "full_model"
folder = "meta_data"

class StackingEnsemble:
    def __init__(self, base_models, meta_model, excluded_cols = None, n_folds=5, name = ""):
        self.base_models = base_models
        self.__meta_model = meta_model
        self.n_folds = n_folds
        self.__fitted_base_models = None
        self.__trained = False
        self.__path = folder
        if name == "":
            self.__model_name = model_name
        else:
            self.__model_name = name
        self.__excluded_cols = excluded_cols

        self.__fold_scores = {}

    def fit(self, X, y, save = True):

        # 1. Stratified K-fold
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state= 42)
        # 2. Train base models per fold

        self.__fold_scores = {name: [] for name in self.base_models.keys()}

        # X_train, y_train, x_test, y_test = train_test_split(X, y, stratify = y)
        n_samples = X.shape[0] # is this not the same as len(X)?

        # amount_models = len(self.base_models.keys())
        amount_models = len(self.base_models)


        # rows = samples, columns = models
        oof_preds = np.zeros((n_samples, amount_models))
        oof_true = np.zeros(n_samples)

        
        for fold, (train_index, value_index) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{self.n_folds}")

            X_train, X_validate = X.iloc[train_index], X.iloc[value_index]
            y_train, y_validate = y.iloc[train_index], y.iloc[value_index]

            for m, (name, model) in enumerate(self.base_models.items()):

                # Since we did base_models, this is essential! (otherwise will only have the same model).
                model_k = clone(model)
                model_k.fit(X_train, y_train)

                # store probability for class "1" (TDE)
                # oof_preds[value_index, m] = model_k.predict_proba(X_validate)[:, 1]
                oof_preds[value_index, m] = self.__get_p1_proba(model_k, X_validate)
                oof_true[value_index] = y_validate.values
                self.__fold_scores[name].append(roc_auc_score(y_validate, self.__get_p1_proba(model_k, X_validate)))



        
        oof_df = pd.DataFrame(
            oof_preds,
            columns=list(self.base_models.keys())
        )

        oof_df["y_true"] = oof_true

        path = os.path.join(self.__path, "oof_probs.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        oof_df.to_csv(path, index=False)


        # 4. Train meta-model on OOF
        self.__meta_model.fit(oof_preds, y)

        # 5. Retrain base models on full train/dev
        self.__fitted_base_models = {}

        for name, model in self.base_models.items():
            final_model = clone(model)
            final_model.fit(X, y)
            self.__fitted_base_models[name] = final_model
            print(f"Final model trained: {name}")
        

        self.__trained = True

        self._oof_preds = oof_preds
        self._oof_true = oof_true


        if save:
            self.save_model()


    @property
    def check_trained(self):
        return self.__trained
    

    @property
    def get_excluded_cols(self):
        return self.__excluded_cols
    

    @property
    def lest_order_feature_importances(self):
        if not self.check_trained:
            print("Needs to train first!")
            raise ValueError
        elif "rfc" not in self.__fitted_base_models.keys():
            print("No random forrest in base models")
            raise KeyError
        
        # importances =  self.__fitted_base_models['rfc'].feature_importances_
        # indices = np.argsort(importances)

        # least_important_feature = X_current.columns[indices[0]]
        
        return np.argsort(self.__fitted_base_models['rfc'].feature_importances_)


    def predict_proba(self, X):
        if not self.check_trained:
            print("You need to train first")
            raise ValueError
        
        if self.__excluded_cols:
            X = X.drop(columns=self.__excluded_cols, errors="ignore")

        try:
            base_probs = np.column_stack([
                # model.predict_proba(X)[:, 1]
                self.__get_p1_proba(model, X)
                for model in self.__fitted_base_models.values()
            ])

            return self.__meta_model.predict_proba(base_probs)
        
        except Exception as e:
            print(f"Meta-model failed, using null safe models as fallback ({e})")

            p1 = self.__null_meta_fallback(X)
            return np.column_stack([1 - p1, p1])
        
        # except ValueError:
        #     print("Error, can't include all, proceed with NAN safe base_models")
        #     return self.__get_p1_proba(self.__fitted_base_models['xgb'], X)
        #     # return self.__fitted_base_models['xgb'].predict_proba


    def save_model(self):
        path = os.path.join(self.__path, self.__model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump(self, path)
        print(f"Model saved to {path}")


    @property
    def name(self):
        return self.__model_name

    @name.setter
    def set_name(self, name):
        if len(name) > 0:
            self.__model_name = name
        else:
            raise ValueError("model_name must be a non-empty string")


    def __null_meta_fallback(self, X):
        """
        Fallback ensemble using mean probability of XGB and RF
        """
        required = ["xgb", "rfc"]
        missing = [m for m in required if m not in self.__fitted_base_models]

        if missing:
            raise KeyError(f"Fallback requires models {missing}, but they are missing.")

        p_xgb = self.__get_p1_proba(self.__fitted_base_models["xgb"], X)
        p_rf  = self.__get_p1_proba(self.__fitted_base_models["rfc"], X)

        p1 = 0.5 * (p_xgb + p_rf)

        return p1

    
    @classmethod
    # def load_or_create(cls, name = "saved_model"): #, **init_kwargs):
    def load_or_create(cls, name = "full_model", **init_kwargs):
        """
        Load a fitted model from disk if it exists,
        otherwise create a new (unfitted) instance.
        """
        path = os.path.join(folder, name)
        if os.path.exists(path):
            print(f"Loading model from {path}")
            return load(path)
        
        if not init_kwargs:
            raise FileNotFoundError(
                f"No saved model found at {path}, and no init arguments were provided."
            )
        
        print("No saved model found. Creating new instance.")
        return cls(**init_kwargs)
        # raise FileNotFoundError


    def base_model_predict(self, input_vector) ->dict:
        if not self.check_trained:
            print("You need to train first")
            raise ValueError


        if self.__excluded_cols:
            input_vector = input_vector.drop(columns=self.__excluded_cols, errors="ignore")

        preds = {}
        try:
            for name, model in self.__fitted_base_models.items():
                preds[name] = model.predict(input_vector)

            
        except Exception as e:
            print(f"Meta-model failed, using null safe models as fallback ({e})")

            preds = {}
            for name in "xgb", "rfc":
                preds[name] = self.__fitted_base_models[name].predict(input_vector)

        return preds


    def predict(self, input_vector):
        if not self.check_trained:
            print("You need to train first")
            raise ValueError

        if self.__excluded_cols:
            input_vector = input_vector.drop(columns=self.__excluded_cols, errors="ignore")

        try:
            base_probs = np.column_stack([
                self.__get_p1_proba(model, input_vector)
                for model in self.__fitted_base_models.values()
            ])
            # base_probs = np.column_stack([
            #     model.predict_proba(input_vector)[:, 1]
            #     for model in self.__fitted_base_models.values()
            # ])

            return self.__meta_model.predict(base_probs)
        
        except Exception as e:
            print(f"Meta-model failed, using null safe models as fallback ({e})")


            p1 = self.__null_meta_fallback(input_vector)
            # return np.column_stack([1 - p1, p1])
            return (p1 >= 0.7).astype(int)

    

    def __get_p1_proba(self, model, X):
        """
        Return P(class=1) for a base model, robust to 1D/2D outputs
        """
        proba = model.predict_proba(X)

        # Case 1: model returns class probabilities only (e.g. [p0, p1])
        if proba.ndim == 1:
            # If length == number of samples, we're good
            if len(proba) == len(X):
                return proba

            # Otherwise, this is per-class output → extract P(class=1)
            if hasattr(model, "classes_") and 1 in model.classes_:
                idx = list(model.classes_).index(1)
                return np.full(len(X), proba[idx])
            else:
                return np.zeros(len(X))

        # Case 2: normal (n_samples, n_classes)
        idx = list(model.classes_).index(1)
        return proba[:, idx]


    def evaluate_base_models(self) -> pd.DataFrame:
        """
        Evaluate base model complementarity using OOF predicted probabilities.
        Returns the correlation matrix of probability outputs.
        """
        if not hasattr(self, "_oof_preds"):
            raise ValueError("OOF predictions not found. Train the model first.")

        corr = np.corrcoef(self._oof_preds.T)

        return pd.DataFrame(
            corr,
            index=self.base_models.keys(),
            columns=self.base_models.keys()
        )



    def _scores_per_model(self) -> dict:
        """
        Returns model-wise ROC-AUC scores for each fold.
        """
        if not self.__fold_scores:
            raise ValueError("No fold-scores saved.")

        model_names = list(self.__fold_scores.keys())
        return {m: self.__fold_scores[m] for m in model_names}

    @property
    def friedman(self):
        """
        Performs Friedman test on ROC-AUC scores for each fold.
        """
        scores_by_model = self._scores_per_model()

        # Check that all models have the same number of folds
        lengths = {len(v) for v in scores_by_model.values()}
        if len(lengths) != 1:
            raise ValueError(f"Different amount of fold for each model: { {k: len(v) for k,v in scores_by_model.items()} }")

        stat, p = friedman_test(*scores_by_model.values())
        return stat, p


    @property
    def posthoc_nemenyi(self):
        """
        Performs a post-hoc Nemenyi test on fold-wise ROC-AUC scores.
        """
        scores_by_model = self._scores_per_model()
        model_names = list(scores_by_model.keys())

        # shape: (n_folds, n_models)
        data = np.column_stack([scores_by_model[m] for m in model_names]).astype(float)

        pvals = sp.posthoc_nemenyi_friedman(data)
        pvals.index = model_names
        pvals.columns = model_names
        return pvals