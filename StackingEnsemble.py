
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
# import time, multiprocessing



class StackingEnsemble:
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    # def fit(self, X, y, stratify):
    def fit(self, X, y):

        # X_train, y_train, x_test, y_test = train_test_split(X, y, stratify = stratify)
        # 1. Stratified K-fold
        skf = StratifiedKFold(self.n_folds, shuffle=True)
        # 2. Train base models per fold
        amount_models = len(self.base_models.keys())
        for i, model in enumerate(self.base_models):
            self.base_models[model].fit(X, y)
            print(f"Model{i + 1}/{amount_models}: {model} finished training!")

        # 3. Store OOF predictions
        # 4. Train meta-model on OOF
        # 5. Retrain base models on full train/dev
        pass

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



    def predict_proba(self, X):
        # 1. Get base model probabilities
        # 2. Feed to meta-model
        # 3. Return final probabilities
        pass

    def evaluate_base_models(self):

        pass
