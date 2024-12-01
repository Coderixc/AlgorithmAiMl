import numpy as np
import pandas as pd


class StockDataGenerator():

    def __init__(self):
        columns = ['Date','Open', 'High', 'Low', 'Close']
        self.__df = pd.DataFrame(columns=columns)

    def TuneMyData(self,

                       StartingPrice: float | None = 100,
                       Variance: float | None = 0.02,
                       start_date: str | None = '2024-01-01',
                       Runif: int|None= 10):
            """
            Generate stock data (OHLC) based on input parameters (StartingPrice, Variance, Runif, start_date).

            Parameters:
            - StartingPrice (float): The initial price of the stock.
            - Variance (float): The daily variance (volatility) for stock price movement.
            - Runif (int): The number of trading days to simulate.
            - start_date (str): The start date for the stock data generation (default '2024-01-01').

            Returns:
            - pd.DataFrame: Simulated stock data with OHLC (Open, High, Low, Close).
            """
            # Set default values for StartingPrice and Variance if not provided
            if StartingPrice is None:
                StartingPrice = 100
            if Variance is None:
                Variance = 0.02

            n_days = Runif

           #Normal Distribution (Mu, Sigma**2) -- >  (loc = mu , Scale=Sigma , Size = no of data point required )
            price_changes = np.random.normal(loc=0, scale=Variance, size=n_days)  # Random daily percentage changes
            prices = [StartingPrice]  # Initial price

            # geometric Brownian motion
            for change in price_changes:
                #P_new = P_old * (1 + change)
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)

            # Create a date range for the stock prices (business days)
            date_range = pd.date_range(start=start_date, periods=n_days + 1, freq='B')  # Business days
            self.__df = pd.DataFrame(prices, index=date_range,columns=['Close'])

            # Create predefined Date column for better clarity
            self.__df['Date'] = self.__df.index

            # Generate OHLC data based on Close prices
            self.__df['Open'] = self.__df['Close'].shift(1)  # Open price is the previous day's close
            self.__df['High'] = self.__df[['Open', 'Close']].max(axis=1)  # High is the max of Open and Close
            self.__df['Low'] = self.__df[['Open', 'Close']].min(axis=1)  # Low is the min of Open and Close

            # The first day's Open should be the same as the Close (no prior data)
            self.__df.iloc[0, self.__df.columns.get_loc('Open')] = self.__df.iloc[0]['Close']

            # Drop NaN values that might be created during shifting
            df = self.__df.dropna()

            # Reorder columns to ensure 'Date' is the first column
            self.__df = self.__df[['Date', 'Open', 'High', 'Low', 'Close']]


    def GetOHLCData(self):
        return self.__df





