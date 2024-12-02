from sklearn.metrics import accuracy_score, mean_squared_error

import DataGenerator.OHLCGenerator as Fd
import pandas as pd
from sklearn.svm import SVC, SVR

from sklearn.model_selection import train_test_split

#https://chatgpt.com/c/674c89be-9040-8011-b7dc-0aecdee08435
class SVM():
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

    def __init__(self,df_OHLC: pd.DataFrame):
        # Validate df is a Pandas DataFrame
        if not isinstance(df_OHLC, pd.DataFrame):
            raise TypeError("Input 'df' must be a pandas DataFrame.")

        self.df= df_OHLC

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
        """
        Train the SVM model.
        """
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the SVM model.
        """
        predictions = self.model.predict(X_test)
        if isinstance(self.model, SVC):
            accuracy = accuracy_score(y_test, predictions)
            print(f"Model Accuracy: {accuracy:.2f}")
            return accuracy
        elif isinstance(self.model, SVR):
            rmse = mean_squared_error(y_test, predictions, squared=False)
            print(f"Model RMSE: {rmse:.2f}")
            return rmse

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        return self.model.predict(X)

    def Run(self):
        X_train, X_test, y_train, y_test = self.Preprocess()

        # Step 4: Train Model
        self.train(X_train, y_train)

        # Step 5: Evaluate Model
        self.evaluate(X_test, y_test)

        # Step 6: Predict Future Values
        predictions = self.predict(X_test)
        print("Predictions:", predictions)
