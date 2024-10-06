from sklearn.svm import SVC
from skopt import space

class SVM_Classifier:
    isFit = False
    randomState = 42
    svm = SVC(random_state=randomState)

    # Define parameter space for SVM
    param_space_SVM = {
        'C': space.Real(1e-6, 1e+6, prior='log-uniform'),      # Regularization parameter
        'gamma': space.Real(1e-6, 1e+1, prior='log-uniform'),  # Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
        'kernel': space.Categorical(['linear', 'rbf', 'poly', 'sigmoid']),  # Type of kernel to use
        'degree': space.Integer(2, 5),                        # Degree of the polynomial kernel (only relevant for 'poly')
    }

    def __init__(self,randomState):
        self.randomState = randomState

    def get_param_space(self):
        return self.param_space_SVM
    
    def get_model(self):
        return self.svm
    
    def set_model(self,model):
        if isinstance(model, SVC):
         # Set the model
         self.model = model
        else:
         print("The model is not an instance of SVC.")
         raise ValueError("Provided model is not a SVC instance.")
    
    