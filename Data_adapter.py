import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split

class DataAdapter:
    X = None
    y = None
    X_train = None
    y_train = None
    X_val = None
    y_val = None
    X_test = None
    y_test = None
    X_train_val = None
    y_train_val = None

    randomState = 42

    hasSplittedDataSet = False

    def __init__(self,randomState):
        self.randomState = randomState

    def load_data(self, path):
        # Load MATLAB file if(path extists)
        mat = scipy.io.loadmat(path)
        print(mat)
        X = mat['X']
        X_dense = X.todense()
        Y = mat['Y'].ravel()
    
        df_X = pd.DataFrame(X_dense)
        df_Y = pd.DataFrame(Y)
        self.X = df_X
        self.y = df_Y
        print(self.X.T.shape)  # Should be (57173, num_features)
        print(self.y.shape)  # Should be (57173,) or similar


    def make_train_test_val_split(self, sizeValidationData, sizeTestData):
        assert(sizeTestData <= 1 or sizeTestData <= 1 )
        # First, split into train/validatio and remaining data (tes)
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X.T, self.y, test_size=sizeTestData,random_state=self.randomState) #, random_state=RANDOMSTATE

        # Now split the train/validation data into validation and train sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=sizeValidationData,random_state=self.randomState) #, random_state=RANDOMSTATE

        print(self.X_train.shape)
        print(self.y_train.shape)
        print(self.X_train_val.shape)
        print(self.y_train_val.shape)
        print(self.X_val.shape)
        print(self.y_val.shape)
        print(self.X_test.shape)
        print(self.y_test.shape)
        # Result:
        # - X_train, y_train for training
        # - X_val, y_val for validation
        # - X_test, y_test for final testing
        self.hasSplittedDataSet=True

    def get_X(self):
        return self.X
    
    def get_y(self):
        return self.y
    
    def get_X_train(self):
        return self.X_train
    
    def get_y_train(self):
        return self.y_train

    def get_X_val(self):
        return self.X_val
    
    def get_y_val(self):
        return self.y_val
    
    def get_X_test(self):
        return self.X_test
    
    def get_y_test(self):
        return self.y_test
    
    def get_X_train_val(self):
        return self.X_train_val
    
    def get_y_train_val(self):
        return self.y_train_val



    #Ensure that train-test-validation-split is correct
#assert(X_train.shape ==(7000, 57173))
#assert(X_val.shape ==(1500, 57173))
#assert(X_test.shape ==(1500, 57173))
   

