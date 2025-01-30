import pandas as pd
import numpy as np
from src import BaseAIModel

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class KNNModel(BaseAIModel):

    def __init__(self,df_OHLC: pd.DataFrame,k_neighbours):
        self.df = df_OHLC
        self.k_neigbours = k_neighbours
        self.model = KNeighborsClassifier(n_neighbors= self.k_neigbours)
        self.prepare_features()

    def calculate_rsi(self,series, period): #series  is 1D
        try:
            delta = series.diff(1)
            gain  =  (delta.where(delta > 0,0)).rolling(window=period).mean()
            loss = (delta.where(delta < 0,0)).rolling(window=period).mean()
            rs = gain/loss
            return 100- (100 / 1+rs)

        except Exception as e:
            raise Exception('failed to calculate rsi ..')

    def prepare_features(self):
        try:
            #add rsi
            self.df['sma_10'] = self.df['close'].rolling(window=10).mean()
            self.df['sma_20'] = self.df['close'].rolling(window=20).mean()
            self.df['rsi'] = self.calculate_rsi(self.df['Close'],14)
            self.df['Target'] = np.where(self.df['Close'].shift(-1) > self.df['Close'],1,0)

        except Exception as e:
            raise Exception('failed to add features on data..')

    def load_data(self):
        X_train, X_test , y_train,y_test =None,None,None,None


        # Decide features from entire data
        features = ['Open', 'High', 'Low', 'Close']
        X = self.df[features]
        y = self.df['Target']

        return train_test_split(X, y, test_size=0.2, shuffle=False)

    def train_and_evaluate(self):
        """ divide the data into two parts from learning and testing """
        try:
            X_train , X_test , y_train, y_test= self.load_data()



        except Exception as e:
            raise Exception('failed to divide data for train and test')

