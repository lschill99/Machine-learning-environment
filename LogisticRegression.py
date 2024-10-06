from skopt import space
from sklearn.linear_model import LogisticRegression

class Logistic_Regression_Classifier:
    isFit = False
    randomState = 42
    logistic_regression = LogisticRegression()

    # Define the parameter space for Logistic Regression
    param_space_LogReg = {
        'C': (1e-6, 1e+6, 'log-uniform'),  # Regularization strength
        'solver': ['liblinear', 'saga'],   # Solvers that work with small datasets
        'penalty': ['l1', 'l2'],            # Regularization types
        'max_iter': space.Integer(200,500)
    }

    def __init__(self,randomState):
        self.randomState = randomState

    def get_param_space(self):
        return self.param_space_LogReg
    
    def get_model(self):
        return self.logistic_regression
    
    def set_model(self,model):
        if isinstance(model, LogisticRegression):
         # Set the mode
         self.model = model
        else:
         print("The model is not an instance of LogisticRegression.")
         raise ValueError("Provided model is not a LogisticRegression instance.")