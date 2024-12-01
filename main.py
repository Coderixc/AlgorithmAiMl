import DataGenerator.OHLCGenerator as Fd  #finance data


def ConsoleModel():
    try:
        #__prompt="Chosse Model"

        print("Select Model")
        print("Press 1 : SVM")
        print("Press ii : SVM")


        algorithmMapping = int(input())
        if algorithmMapping==1:
            print("Selected SVM Model..")

    except Exception as e:
        print(F"Error Occured..{e}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stock_gen = Fd.StockDataGenerator()

    # Generate stock data with custom parameters and a start date
    stock_gen.TuneMyData(StartingPrice=150, Variance=0.02, Runif=1000, start_date='2020-02-01')

    # Display the generated OHLC stock data with the Date column
    print(stock_gen.GetOHLCData())

    ConsoleModel()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
