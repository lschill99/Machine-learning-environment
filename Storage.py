class ParameterStorage:
    parameterDecisionTree = []
    parameterLogisticRegression = []
    parameterSVM = []
    parameterRandomForest = []

    best_precision_DecisionTree = 0
    best_precision_LogisticRegression = 0 
    best_precision_SVM = 0
    best_precision_RandomForest = 0

    best_DecisionTree = None
    best_LogisticRegression = None
    best_SVM = None
    best_RandomForest = None

    def set_best_parameters_DecisionTree(self, precision,param, model):
        if(precision > self.best_precision_DecisionTree):
            self.best_precision_DecisionTree = precision
            self.parameterDecisionTree = param
            self.best_DecisionTree = model


    def set_best_parameters_LogisticRegression(self, precision,param, model):
        if(precision > self.best_precision_DecisionTree):
            self.best_precision_LogisticRegression = precision
            self.parameterLogisticRegression = param
            self.best_LogisticRegression = model

    def set_best_parameters_SVM(self, precision,param, model):
        if(precision > self.best_precision_SVM):
            self.best_precision_SVM = precision
            self.parameterSVM = param
            self.best_SVM = model

    def set_best_parameters_RandomForest(self, precision,param, model):
        if(precision > self.best_precision_RandomForest):
            self.best_precision_RandomForest = precision
            self.parameterRandomForest = param
            self.best_RandomForest = model
    
    def get_best_model(self):
        # Create a dictionary to map model names to their precision values
        precisions = {
            "Decision Tree": self.best_precision_DecisionTree,
            "Logistic Regression": self.best_precision_LogisticRegression,
            "SVM": self.best_precision_SVM,
            "Random Forest": self.best_precision_RandomForest,
        }
        # Find the model with the highest precision
        best_model = max(precisions, key=precisions.get)
        best_precision = precisions[best_model]
        # Print the results
        print(f"The model closest to 1 is: {best_model} with a precision for ham of {best_precision:.4f}")
        return best_model