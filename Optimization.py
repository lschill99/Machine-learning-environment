from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV

class ParameterOptimizerBuilder():
    def __init__(self, model, search_type='bayes'):
        self.model = model
        self.search_type = search_type #search_type must be either 'grid' or 'bayes'
        self.param_space = {}
        self.cv = 5
        self.scoring = 'precision'
        self.n_iter = 50  # For Bayesian search

    def set_param_space(self, param_space):
        self.param_space = param_space

    def set_cv(self, cv):
        self.cv = cv

    def set_scoring(self, scoring):
        self.scoring = scoring

    def set_search_type(self, search_type):
        if search_type not in ['grid', 'bayes']:
            raise ValueError("search_type must be either 'grid' or 'bayes'")
        self.search_type = search_type

    def build_search(self):
        if self.search_type == 'grid':
            return GridSearchCV(self.model, self.param_space, cv=self.cv, scoring=self.scoring)
        elif self.search_type == 'bayes':
            return BayesSearchCV(self.model, self.param_space, n_iter=self.n_iter, cv=self.cv, scoring=self.scoring)
        else:
            raise ValueError("Invalid search type set. Please use 'grid' or 'bayes'.")

   