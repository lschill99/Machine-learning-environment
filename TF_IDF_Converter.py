from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

class TF_IDF_Converter:
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

    def __init__(self, X , y, randomState):
        self.X = self.transform_data_from_bow_to_TF_IDF(X)
        self.y = y
        self.randomState = randomState

    def make_train_test_val_split(self, sizeValidationData, sizeTestData):
        assert(sizeTestData <= 1 or sizeTestData <= 1 )
        # First, split into train/validatio and remaining data (tes)
        self.X_train_val, self.X_test, self.y_train_val, self.y_test = train_test_split(self.X.T, self.y, test_size=sizeTestData, random_state=self.randomState) #random state has to be the same as in the data adapter

        # Now split the train/validation data into validation and train sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train_val, self.y_train_val, test_size=sizeValidationData, random_state=self.randomState) #random state has to be the same as in the data adapter
        print('TF-IDF Converter has the following partitions:')
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

    def transform_data_from_bow_to_TF_IDF(self, X_data):
        # Convert BoW to a sparse matrix (which is more efficient for large data)
        # Assuming df_X is your DataFrame with 10,000 rows and 57,173 columns
        # representing the BoW of emails

        # Step 1: Convert DataFrame to a sparse matrix (if it's not already sparse)
        BoW_sparse = csr_matrix(X_data.values)

        # Step 2: Initialize the TfidfTransformer
        tfidf_transformer = TfidfTransformer()

        # Step 3: Fit and transform the sparse BoW matrix into TF-IDF
        X_tfidf = tfidf_transformer.fit_transform(BoW_sparse)

        # The result is a sparse matrix in TF-IDF format
        # To view the TF-IDF matrix as a dense matrix (optional):
        tfidf_dense = X_tfidf.toarray()

        # If you want to inspect the IDF values (optional)
        idf_values = tfidf_transformer.idf_

        # Print the shape of the TF-IDF matrix
        print(f"TF-IDF matrix shape: {X_tfidf.shape}")
        return X_tfidf

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
