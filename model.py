from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score


class Model:  #This class is deprecated and is of no further use but could be redesigned to become an interface
    
    def __init__(self, model):
        self.model = model
        self.accuracy = None
        self.f1 = None
        self.recall = None
        self.precision = None
    
    def calc_metrics(self, y_test, y_pred):
        self.accuracy = accuracy_score(y_test, y_pred)
        # Using macro to handle multiclass classification as well
        self.precision = precision_score(y_test, y_pred, average='macro')  # Use 'macro' for multiclass
        self.recall = recall_score(y_test, y_pred, average='macro')
        self.f1 = f1_score(y_test, y_pred, average='macro')    
        
    #acc = number of correct pred / all = (TP + TN) / (TP + TN + FP + FN)
    #recall = TP / TP+ FN
    #precision = TP/TP+FP
    #f1 score = 2* Recall * Precision / (Recall + Precision)  
    def perform(self, X_train, y_train, X_test, y_test):
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.calc_metrics(y_test,y_pred)
        
        
    def __str__(self):  # Correcting the method
        return (f"Accuracy: {self.accuracy}\n"
                f"Precision: {self.precision}\n"
                f"Recall: {self.recall}\n"
                f"F1 Score: {self.f1}")