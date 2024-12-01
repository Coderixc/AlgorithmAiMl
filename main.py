import numpy as np
import pandas as pd

class StockDataGenerator():

    def __init__(self):
        columns = ['Open', 'High', 'Low', 'Close']
        self.__df = pd.DataFrame(columns=columns)


    def TuneMyData(self,StartingPrice : int | None ,
                   Variance = float | None,
                   Runif = int | 10):

        if StartingPrice is None:
            StartingPrice = 100

        if Variance is None:
            Variance = 0.02

        n_days = Runif


        price_changes = np.random.normal(loc=0, scale=Variance, size=n_days)
        prices = [StartingPrice]  # Initial price


        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        date_range = pd.date_range(start='2024-01-01', periods=n_days + 1, freq='B')  # Business days
        df = pd.DataFrame(prices, index=date_range, columns=['Close'])

        # Generate OHLC data based on Close prices
        df['Open'] = df['Close'].shift(1)  # Open price is the previous day's close
        df['High'] = df[['Open', 'Close']].max(axis=1)  # High is the max of Open and Close
        df['Low'] = df[['Open', 'Close']].min(axis=1)  # Low is the min of Open and Close

        # The first day's Open should be the same as the Close (no prior data)
        df.iloc[0, df.columns.get_loc('Open')] = df.iloc[0]['Close']

        # Drop NaN values that might be created during shifting
        df = df.dropna()

        return df



    def GetOHLCData(self):
        return self.__df




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
