from skopt import space
from sklearn.tree import DecisionTreeClassifier



class Tree_Classifier:

    isFit = False
    randomState = 42
    decision_tree = DecisionTreeClassifier(random_state=randomState)

    param_space_DecisionTree = {
    'criterion': space.Categorical(['gini', 'entropy']),  # The function to measure the quality of a split
    'max_depth': space.Integer(1, 50),                     # Maximum depth of the tree
    'min_samples_split': space.Integer(2, 10),             # Minimum number of samples required to split an internal node
    'min_samples_leaf': space.Integer(1, 10),              # Minimum number of samples required to be at a leaf node
    'max_features': space.Categorical([None, 'sqrt', 'log2']),  # Number of features to consider when looking for the best split
    }

    def __init__(self,randomState):
        self.randomState = randomState

    def get_param_space(self):
        return self.param_space_DecisionTree
    
    def get_model(self):
        return self.decision_tree
    
    def set_model(self,model):
        if isinstance(model, DecisionTreeClassifier):
         print("The model is an instance of DecisionTreeClassifier.")
         # Set the model 
         self.model = model
        else:
         print("The model is not an instance of DecisionTreeClassifier.")
         raise ValueError("Provided model is not a DecisionTreeClassifier instance.")