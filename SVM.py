
from sklearn.metrics import accuracy_score, mean_squared_error

import DataGenerator.OHLCGenerator as Fd
import pandas as pd
from  sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, SVR

from sklearn.model_selection import train_test_split
import BaseAIModel as baseAI

class KernelFunc:
    LINEAR = "linear"
    POLY = "poly"
    RBF = "rbf"
    SIGMOID = "sigmoid"
    PRECOMPUTED = "precomputed"

class SVM(baseAI.BaseAIModel):
    """
    Support Vector Machines (SVM) are supervised learning models
    Used - classification and regression problems.
    Aim - find an optimal hyperplane in a multi-dimensional space
        that separates the data into classes (classification)
        or fits a relationship (regression).

    Key Concepts
    Hyperplane: A decision boundary that separates classes in the feature space.
    Support Vectors: Data points that are closest to the hyperplane and influence its position.
    Margin: The distance between the hyperplane and the nearest data points (support vectors).
    SVM maximizes this margin.

    Distance between Margin and Huperplane : we use Euclidean distance
    """

    def __init__(self,df_OHLC: pd.DataFrame, kernel= KernelFunc.LINEAR):
        # Validate df is a Pandas DataFrame
        if not isinstance(df_OHLC, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")
        self.df= df_OHLC

        if kernel == KernelFunc.LINEAR:
            self.__kernel = KernelFunc.LINEAR
        elif kernel == KernelFunc.RBF:
            self.__kernel = KernelFunc.RBF
        elif kernel == KernelFunc.POLY:
            self.__kernel = KernelFunc.POLY
        elif kernel == KernelFunc.PRECOMPUTED:
            self.__kernel = KernelFunc.PRECOMPUTED


    def Preprocess(self):
        """
        Preprocess the injected dataframe to create features and Target

        Operation :
        - pct_change = (Current_Value - Prev_value) / Prev_value
        - Direction : 1 for up  , 0 for down
        - Trading Indicator : MA(5) and MA(10)
        """
        df = self.df.copy(deep= True)  #deep = data(independent) + structure
        #add a 'Return' column as daily return
        df['Return'] = df['Close'].pct_change()
        #calculate Direction using True(1) , False(0) , Converting Boolean to int using astype(int)
        df['Direction'] = (df['Return'] > 0).astype(int)
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df = df.dropna()
        if len(df) <=0:
            print("No Data")
            return

        # Define features and target
        #synatax : [[]]  = for selecting multiple columns
        #[] = list type
        #[ list type ]

        #df['Open'] -- series type
        #df[['Open']] -- dataframe type
        X = df[['Open', 'High', 'Low', 'MA5', 'MA10']]  #as dataframe
        y = df['Direction']  #as series
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the SVM model.
        """
        predictions = self.predictModel(X_test)
        if isinstance(self.model, SVC):
            accuracy = accuracy_score(y_test, predictions)
            print(f"Model Accuracy: {accuracy:.2f}")
            return accuracy
        elif isinstance(self.model, SVR):
            rmse = mean_squared_error(y_test, predictions, squared=False)
            print(f"Model RMSE: {rmse:.2f}")
            return rmse

    def predictModel(self, X):
        """
        Make predictions using the trained model.
        """
        return self.model.predict(X)
    def ModelSelection(self,X_train,y_train):
        param_grid= {
            "C": [0.1,1,10],
            'gamma': [1,0.1,0.01],
            'kernel':[KernelFunc.RBF, KernelFunc.LINEAR]
        }
        grid = GridSearchCV(SVC(), param_grid,refit=True,verbose=2 )
        grid.fit(X_train,y_train)
        print("Best Parameters:", grid.best_params_)


    def Run(self , HyperParamater = False ):
        try:
            X_train, X_test, y_train, y_test = self.Preprocess()
            self.model = SVC(kernel=self.__kernel)
            self.train(X_train, y_train)
            self.evaluate(X_test,y_test)
            if HyperParamater == True:
                self.ModelSelection(X_train,y_train)

        except Exception as e:
            print(F"Error Occured {e}")