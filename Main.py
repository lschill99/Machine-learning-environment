import numpy as np

from Data_adapter import DataAdapter
from Optimization import ParameterOptimizerBuilder
from SVM import SVM_Classifier
from TF_IDF_Converter import TF_IDF_Converter

from scipy.sparse import hstack
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from skopt.space import Real, Integer
from sklearn.metrics import precision_score, make_scorer
from skopt import space
from DecisionTree import Tree_Classifier
from LogisticRegression import Logistic_Regression_Classifier
from RandomForest import Random_Forest_Classifier
from Storage import ParameterStorage
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, roc_auc_score



TEST_SIZE = 0.2
VALIDATION_SIZE= 0.2
TRAIN_SIZE= 0.6

RANDOMSTATE = 42
N_SPLITS = 3

SEARCHTYPE = 'bayes'  #search_type must be either 'grid' or 'bayes'
CV = 2
SCORER ='precision'

data_adapter = DataAdapter(RANDOMSTATE)

storage = ParameterStorage()

data_adapter.load_data('data/emails.mat')
tf_idf_Converter = TF_IDF_Converter(data_adapter.get_X(),data_adapter.get_y(),RANDOMSTATE)
data_adapter.make_train_test_val_split(0.125, 0.2)
tf_idf_Converter.make_train_test_val_split(0.125, 0.2)

#Training with Bow & TF-IDF combined -computationally intensive
#X_train_idf_bow = hstack([data_adapter.get_X_train(), tf_idf_Converter.get_X_train()])
#X_val_idf_bow = hstack([data_adapter.get_X_val(), tf_idf_Converter.get_X_val()])
#X_train_val_idf_bow = hstack([data_adapter.get_X_train_val(), tf_idf_Converter.get_X_train_val()])
#X_test_idf_bow = hstack([data_adapter.get_X_test(), tf_idf_Converter.get_X_test()])

if(data_adapter.hasSplittedDataSet):
    print('data has been splitted')

def get_ham_precision_score(y_true,y_pred):
    return precision_score(y_true,y_pred, labels=[-1], zero_division=0)

def optimizerWrapper(model,param_space,cv, scorer, search_type):
    optimizerBuilder = ParameterOptimizerBuilder(model)
    optimizerBuilder.set_param_space(param_space)
    optimizerBuilder.set_cv(cv)
    optimizerBuilder.set_scoring(scorer)
    optimizerBuilder.set_search_type(search_type)
    opt = optimizerBuilder.build_search()


# Initialize the Logistic Regression classifier
logisticreg = Logistic_Regression_Classifier(RANDOMSTATE)
# Initialize the Random Forest classifier
randomForest = Random_Forest_Classifier(RANDOMSTATE)
# Initialize the Decision Tree classifier
decision_tree = Tree_Classifier(RANDOMSTATE)
# Initialize the SVM classifier
svm = SVM_Classifier(RANDOMSTATE)



def parameter_tuning(param_space, model,X_train, y_train,X_val,y_val):
    model.fit(X_train,y_train[0])
    # BayesSearchCV for Bayesian optimization   
    opt = BayesSearchCV(
        estimator=model,
        search_spaces=param_space,
        n_iter=20,  # Number of iterations for optimization
        cv=3,       # 3-fold cross-validation
        scoring= make_scorer(get_ham_precision_score),
        #n_jobs=-1,  # Use all CPU cores
        random_state= RANDOMSTATE
    )
    # Fit the model
    opt.fit(X_val, y_val[0])
    
    # Print the best hyperparameters and corresponding accuracy score
    print("Best hyperparameters: ", opt.best_params_)
    print("Best cross-validation precision for ham: {:.4f}".format(opt.best_score_))
    
    # Evaluate on the validation data
    test_precision_ham = opt.score(X_val, y_val[0])
    print(f"Test precision ham on validation data: {test_precision_ham :.4f}")
    return opt

#Method for using grid search or bayes search based on SEARCHTYPE - currently not in use
def parameter_tuning_with_FLEX_opt(classifier,X_train, y_train,X_val,y_val):
    model = classifier.get_model()
    model.fit(X_train,y_train[0])
    param_space= classifier.get_param_space()

    opt = optimizerWrapper(model,param_space,CV,SEARCHTYPE)
    # Fit the model
    opt.fit(X_val, y_val[0])
    
    # Print the best hyperparameters and corresponding accuracy score
    print("Best hyperparameters: ", opt.best_params_)
    print("Best cross-validation accuracy: {:.2f}".format(opt.best_score_))
    
    # Evaluate on the test data
    test_accuracy = opt.score(X_val, y_val[0])
    print(f"Test accuracy: {test_accuracy:.2f}")
    return opt

#Testing the best model provided by the optimizer on the test data
def test_best_model_on_test_data(X_train_val, y_train_val,X_test,y_test, optimizer):
        best_params = optimizer.best_params_
        best_model = optimizer.best_estimator_
        best_model.fit(X_train_val, y_train_val)
        y_predictions_X_test = best_model.predict(X_test)
        precision_score_best_model = get_ham_precision_score(y_test,y_predictions_X_test)
        print(f'Precision for ham of the best tuned model on the test data:  {precision_score_best_model :.4f}')
        return precision_score_best_model,best_params,best_model

def plot_roc_curve(y_true, predictions):
    # Compute FPR, TPR, and thresholds for ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, predictions) #,pos_label=[-1]
    # Calculate AUC-ROC score
    roc_auc = roc_auc_score(y_true, predictions) #labels=[-1]

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')  # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(y_true, predictions):
    # Compute Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, predictions)

    # Compute the average precision score
    avg_precision = average_precision_score(y_true, predictions)

    # Plot the Precision-Recall curve
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {avg_precision:.4f})', color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show() 

def calc_metrics(classifier, X_data, y_data):
        if(not classifier.isFit):
            print('classifier was not fitted')
            return 0,0,0,0
        else:
           predictions = classifier.get_model().predict(X_data)
           report = classification_report(y_data,predictions, digits = 4,output_dict=True)
           precision_ham = report['-1']['precision']
           recall_ham = report['-1']['recall']
           f1_score_ham = report['-1']['f1-score']
           accuracy = accuracy_score(y_data, predictions)
           precision = precision_score(y_data,predictions)  # Use 'macro' for multiclass
           recall = recall_score(y_data, predictions, average='macro')
           f1 = f1_score(y_data, predictions, average='macro')
           print(f"Accuracy: {accuracy}\n"
                f"Precision: {precision}\n"
                f"Recall: {recall}\n"
                f"F1 Score: {f1}\n"
                f"_________HAM SECTION__________\n"
                f"Precision ham: {precision_ham}\n"
                f"Recall ham: {recall_ham}\n"
                f"F-1 score ham: {f1_score_ham}\n")
           
           plot_roc_curve(y_data,predictions)
           plot_precision_recall_curve(y_data,predictions)
           return(accuracy,precision,recall,f1)

ham_precision_scorer = make_scorer(get_ham_precision_score)





print('________Logistic Regression________')
optLogReg = parameter_tuning(logisticreg.get_param_space(),logisticreg.get_model(), data_adapter.get_X_train(), data_adapter.get_y_train(), data_adapter.get_X_val(), data_adapter.get_y_val())
precision_score_best_model,best_params,best_model = test_best_model_on_test_data(data_adapter.get_X_train_val(),data_adapter.get_y_train_val(),data_adapter.get_X_test(),data_adapter.get_y_test(),optLogReg)
storage.set_best_parameters_LogisticRegression(precision_score_best_model,best_params,best_model)
storage.get_best_model()
logisticreg.set_model(best_model)
logisticreg.isFit = True
m = calc_metrics(logisticreg,data_adapter.get_X_test(),data_adapter.get_y_test())



print('________RandomForest________')
optRandomForest = parameter_tuning(randomForest.get_param_space(),randomForest.get_model(),data_adapter.get_X_train(), data_adapter.get_y_train(), data_adapter.get_X_val(), data_adapter.get_y_val())
precision_score_best_model,best_params,best_model = test_best_model_on_test_data(data_adapter.get_X_train_val(),data_adapter.get_y_train_val(),data_adapter.get_X_test(),data_adapter.get_y_test(),optRandomForest)
storage.set_best_parameters_RandomForest(precision_score_best_model,best_params,best_model)
randomForest.set_model(best_model)
randomForest.isFit=True
m = calc_metrics(randomForest,data_adapter.get_X_test(),data_adapter.get_y_test())


print('________Decision Tree________')
optDecisionTree = parameter_tuning(decision_tree.get_param_space(),decision_tree.get_model(), data_adapter.get_X_train(), data_adapter.get_y_train(), data_adapter.get_X_val(), data_adapter.get_y_val())
precision_score_best_model,best_params,best_model = test_best_model_on_test_data(data_adapter.get_X_train_val(),data_adapter.get_y_train_val(),data_adapter.get_X_test(),data_adapter.get_y_test(),optDecisionTree)
storage.set_best_parameters_DecisionTree(precision_score_best_model,best_params,best_model)
storage.get_best_model()
decision_tree.set_model(best_model)
decision_tree.isFit = True
m = calc_metrics(decision_tree,data_adapter.get_X_test(),data_adapter.get_y_test())


print('________SVM________')
optSVM = parameter_tuning(svm.get_param_space(),svm.get_model(),data_adapter.get_X_train(), data_adapter.get_y_train(), data_adapter.get_X_val(), data_adapter.get_y_val())
precision_score_best_model,best_params,best_model = test_best_model_on_test_data(data_adapter.get_X_train_val(),data_adapter.get_y_train_val(),data_adapter.get_X_test(),data_adapter.get_y_test(),optSVM)
storage.set_best_parameters_SVM(precision_score_best_model,best_params,best_model)
storage.get_best_model()
svm.set_model(best_model)
svm.isFit = True
m = calc_metrics(svm,data_adapter.get_X_test(),data_adapter.get_y_test())





