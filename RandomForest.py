import BaseAIModel as baseAI
import pandas as pd

from sklearn.model_selection import train_test_split

class RandomForest(baseAI.BaseAIModel):

    def __ini__(self, df_OHLC: pd.DataFrame):
        if not isinstance(df_OHLC,pd.DataFrame) :
            raise TypeError(F"Input 'df' must be a pandas DataFrame.")

        self.df= df_OHLC
        pass

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
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'HL_PCT', 'CO_PCT', '5_MA', '20_MA']
            X = self.df[features]
            y_class = self.df['Price_Up']  # For classification

            y_reg = self.df['Close'].shift(-1)  # For regression

            return train_test_split(X, y_class, test_size=0.2, random_state=42)



        except Exception as e:
            print(F"Error Ocurred while process random forest {e}")

    def train(self):
        pass

    def evaluate(self):
        pass

    def predictModel(self):
        pass

    def Run(self):
        X_train, X_test, y_train, y_test = self.Preprocess()
        self.model = RandomForest()
        self.train(X_train, y_train)
        self.evaluate(X_test, y_test)
