from src import BaseAIModel as baseAI
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.model_selection import GridSearchCV

class RandomForest(baseAI.BaseAIModel):

    def __init__(self, df_OHLC: pd.DataFrame):
        if not isinstance(df_OHLC,pd.DataFrame) :
            raise TypeError(F"Input 'df' must be a pandas DataFrame.")
        #print("cjeck")

        self.df= df_OHLC
        print(F" {len(self.df)}")
        self.model = RandomForestClassifier(n_estimators=100, random_state= 42)

    def preprocess(self):
        try:
            # Feature engineering
            self.df['HL_PCT'] = (self.df['High'] - self.df['Low']) / self.df['Open'] * 100
            self.df['CO_PCT'] = (self.df['Close'] - self.df['Open']) / self.df['Open'] * 100
            self.df['5_MA'] = self.df['Close'].rolling(window=5).mean()
            self.df['20_MA'] = self.df['Close'].rolling(window=20).mean()

            self.df.dropna(inplace= True)

            self.df['Price_Up'] = (self.df['Close'].shift(-1) > self.df['Close']).astype(int)

            # Features and target
            features = ['Open', 'High', 'Low', 'Close', 'HL_PCT', 'CO_PCT', '5_MA', '20_MA']
            X = self.df[features]
            y_class = self.df['Price_Up']  # For classification

            y_reg = self.df['Close'].shift(-1)  # For regression

            return train_test_split(X, y_class, test_size=0.2, random_state=42)


        except Exception as e:
            print(F"Error Ocurred while process random forest {e}")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self,X_test, y_test):
        """
                Evaluate the SVM model.
                """
        predictions = self.predictModel(X_test)
        if isinstance(self.model, RandomForestClassifier):
            accuracy = accuracy_score(y_test, predictions)
            print(f"Model Accuracy: {accuracy:.2f}")
            return accuracy
        elif isinstance(self.model, RandomForestRegressor):
            rmse = mean_squared_error(y_test, predictions, squared=False)
            print(f"Model RMSE: {rmse:.2f}")
            return rmse

    def predictModel(self,X):
        return self.model.predict(X)

    def modeltunning(self,X_train, y_train):
        """ Test Molde with all possibilities """
        print("Testing and tuning all hyper parameters")
        try:
            param_grid = {
                'n_estimators': [100, 200, 500],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=2)
            print("training model")
            grid_search.fit(X_train, y_train)
            print("Best Parameters:", grid_search.best_params_)

        except Exception as e:
            print(F"Error Generated while tuning all combinations :{e}")

    def Run(self ,HyperParamater = False):
        X_train, X_test, y_train, y_test = self.preprocess()
        self.train(X_train, y_train)
        self.evaluate(X_test, y_test)

        if HyperParamater == True:
            self.modeltunning(X_train, y_train)
