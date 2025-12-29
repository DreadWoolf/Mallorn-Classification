
# Do a meta logistic regression model.
# Even though the labels are binary, we stack using predicted probabilities; 
# the logistic meta-model learns how to weight and combine confidence scores rather than hard votes.
# “Because TDEs are rare, we do not use the default 0.5 decision threshold.
#  Instead, we select the probability cutoff that achieves 90% completeness on the validation set and report the resulting purity on the test set.”

#  Stratified k-fold with 5

# Workflow guide...
#1 Use stratified split to create a held-out test set (do this first).
#2 Use stratified K-fold on the remaining Train/Dev set.
#3 For each fold: train base models on K−1 folds.
#4 Validate on the held-out fold and store probabilities (OOF).
#5 Repeat for all folds to fill OOF probabilities for every train/dev sample.
#6 Train the meta-model on the OOF probability matrix.
#7 Retrain each base model on the full Train/Dev set (not the full dataset including test).
#8 Use retrained base models + meta-model to predict on the held-out test set.

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.base import clone
import numpy as np
import pandas as pd
from joblib import dump, load
import os

# import time, multiprocessing
model_name = "saved_model"
folder = "meta_data"

class StackingEnsemble:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.__meta_model = meta_model
        self.n_folds = n_folds
        self.__fitted_base_models = None
        self.__trained = False
        # self.__path = "meta_data"
        # self.__model_name = "saved_model"
        self.__path = folder
        self.__model_name = model_name
    

    def fit(self, X, y):

        # 1. Stratified K-fold
        skf = StratifiedKFold(self.n_folds, shuffle=True, random_state= 42)
        # 2. Train base models per fold

        # X_train, y_train, x_test, y_test = train_test_split(X, y, stratify = y)
        n_samples = X.shape[0] # is this not the same as len(X)?

        # amount_models = len(self.base_models.keys())
        amount_models = len(self.base_models)


        # rows = samples, columns = models
        oof_preds = np.zeros((n_samples, amount_models))
        oof_true = np.zeros(n_samples)

        
        for fold, (train_index, value_index) in enumerate(skf.split(X, y)):
            print(f"Fold {fold + 1}/{self.n_folds}")

            # # 🔧 FIX: update NN input_dim dynamically
            # if "model__input_dim" in model_k.get_params():
            #     model_k.set_params(model__input_dim=X_train.shape[1])

            # 
            # X_train, X_validate = X[train_index], X[value_index]
            # y_train, y_validate = y[train_index], y[value_index]

            X_train, X_validate = X.iloc[train_index], X.iloc[value_index]
            y_train, y_validate = y.iloc[train_index], y.iloc[value_index]

            for m, (name, model) in enumerate(self.base_models.items()):

                # Since we did base_models, this is essential! (otherwise will only have the same model).
                model_k = clone(model)

                # 🔧 FIX: update NN input_dim dynamically (only if applicable)
                params = model_k.get_params()
                if "model__input_dim" in params:
                    model_k.set_params(model__input_dim=X_train.shape[1])


                model_k.fit(X_train, y_train)


                # store probability for class "1" (TDE)
                oof_preds[value_index, m] = model_k.predict_proba(X_validate)[:, 1]
                oof_true[value_index] = y_validate.values




        # 3. Store OOF predictions

        # save_oof_path="meta_data/oof_probs.csv"
        # # Make directory if needed
        # os.makedirs(os.path.dirname(save_oof_path), exist_ok=True)

        # # Option A: Save only the OOF matrix (NumPy)
        # np.savetxt(save_oof_path, oof_preds, delimiter=",", fmt="%.6f")

        
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
        # for i, model in enumerate(self.base_models):
        #     self.base_models[model].fit(X, y)
        #     print(f"Model {i + 1}/{amount_models}: {model} finished training!")


        # 5. Retrain base models on full train/dev
        self.__fitted_base_models = {}

        for name, model in self.base_models.items():
            final_model = clone(model)
            final_model.fit(X, y)
            self.__fitted_base_models[name] = final_model
            print(f"Final model trained: {name}")
        

        self.__trained = True

        self.save_model()


    # def __parallelism_work(self, target:function, args:tuple):
    #     jobs = []
    #     for model in self.base_models:
    #         out_list = list()
    #         # process = multiprocessing.Process(target=model.fit(X_train, y_train), args=(size, out_list))
    #         process = multiprocessing.Process(target=model.target, args=args)
    #         jobs.append(process)

    #     for j in jobs:
    #         j.start()
    #     for j in jobs:
    #         j.join()


    @property
    def check_trained(self):
        return self.__trained
    # def predict_proba(self, X):
    #     # 1. Get base model probabilities
    #     # 2. Feed to meta-model
    #     # 3. Return final probabilities
    #     pass

    def predict_proba(self, X):
        base_probs = np.column_stack([
            model.predict_proba(X)[:, 1]
            for model in self.__fitted_base_models.values()
        ])

        return self.__meta_model.predict_proba(base_probs)


    def save_model(self):
        path = os.path.join(self.__path, self.__model_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        dump(self, path)
        print(f"Model saved to {path}")

    
    # @property
    @classmethod
    def load_or_create(cls, name = "saved_model"): #, **init_kwargs):
        """
        Load a fitted model from disk if it exists,
        otherwise create a new (unfitted) instance.
        """

        path = os.path.join(folder, name)
        if os.path.exists(path):
            print(f"Loading model from {path}")
            return load(path)
        
        raise FileNotFoundError
        # print("No saved model found. Creating new instance.")
        # return cls(**init_kwargs)


    def base_model_predict(self, input_vector) ->dict:
        if not self.check_trained:
            print("You need to train first")
            raise ValueError

        preds = {}

        for name, model in self.__fitted_base_models.items():
            preds[name] = model.predict(input_vector)


        return preds

    def predict(self, input_vector):
        if not self.check_trained:
            print("You need to train first")
            raise ValueError


        base_probs = np.column_stack([
            model.predict_proba(input_vector)[:, 1]
            for model in self.__fitted_base_models.values()
        ])
        # predictions = []
        # # for model in self.__fitted_base_models:
        # for model in self.__fitted_base_models.values():
        #     # predictions.append(model.predict_proba(input_vector))
        #     predictions.append(model.predict_proba(input_vector)[:, 1])

        return self.__meta_model.predict(base_probs)

    def evaluate_base_models(self):

        pass
