from sklearn.ensemble import RandomForestClassifier
from skopt import space

class Random_Forest_Classifier:
    isFit = False
    randomState = 42
    randomForest = RandomForestClassifier(n_estimators=100, random_state=randomState)

    param_space_RandomForest = {
        'n_estimators': space.Integer(50, 200),              # Number of trees
        'max_depth': space.Integer(2, 30),                   # Maximum depth of the trees
        'min_samples_split': space.Integer(2, 10),           # Minimum number of samples to split a node
        'min_samples_leaf': space.Integer(1, 10),            # Minimum samples per leaf
        'max_features': space.Categorical(['sqrt', 'log2']), # Categorical choice
        'bootstrap': space.Categorical([False, True])        # Whether to use bootstrap sampling
    }
    
    def __init__(self,randomState):
        self.randomState = randomState

    def get_param_space(self):
        return self.param_space_RandomForest
    
    def get_model(self):
        return self.randomForest
    
    def set_model(self,model):
        if isinstance(model, RandomForestClassifier):
         # Set the mode
         self.model = model
        else:
         print("The model is not an instance of RandomForestClassifier.")
         raise ValueError("Provided model is not a RandomForestClassifier instance.")