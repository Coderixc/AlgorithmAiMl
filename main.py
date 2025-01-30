from data  import DataGenerator
from data import DataGenerator as Fd  #finance data
from data.DataGenerator.OHLCGenerator import StockDataGenerator
from src import SVM
from src import RandomForest as rf


def ConsoleModel(df):
    try:
        #__prompt="Chosse Model"
        print("Select Model")
        print("Press 1 : SVM with Different Kernel Tick")
        print("Press  2 : Random Forest")
        print("Press  3 : K-Nearest Neighbors (KNN) ")

        algorithmMapping = int(input())
        if algorithmMapping==1:
            print("Selected SVM Model..")
            svm_model = SVM.SVM(df,kernel=SVM.KernelFunc.LINEAR)
            svm_model.Run(HyperParamater= True)

        elif  algorithmMapping==2:
            print("Selected Random Forest Model..")
            rand_forest = rf.RandomForest(df)
            rand_forest.Run(HyperParamater= True)
        elif  algorithmMapping==3:
            print("K-Nearest Neighbors (KNN)")
            rand_forest = rf.RandomForest(df)
            rand_forest.Run(HyperParamater= True)

    except Exception as e:
        print(F"Error Occured..{e}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stock_gen = StockDataGenerator()

    # Generate stock data with custom parameters and a start date
    stock_gen.TuneMyData(StartingPrice=150, Variance=0.02, Runif=6000, start_date='2000-02-01')
    ConsoleModel(stock_gen.GetOHLCData())


