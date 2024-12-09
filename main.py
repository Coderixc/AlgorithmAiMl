import DataGenerator.OHLCGenerator as Fd  #finance data
import SVM
import RandomForest as rf


def ConsoleModel(df):
    try:
        #__prompt="Chosse Model"

        print("Select Model")
        print("Press 1 : SVM with Different Kernel Tick")
        print("Press  2 : Random Forest")

        algorithmMapping = int(input())
        if algorithmMapping==1:
            print("Selected SVM Model..")
            svm_model = SVM.SVM(df,kernel=SVM.KernelFunc.LINEAR)
            svm_model.Run(HyperParamater= True)

        elif  algorithmMapping==2:
            print("Selected Random Forest Model..")
            rand_forest = rf.RandomForest(df)
            rand_forest.Run()

    except Exception as e:
        print(F"Error Occured..{e}")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    stock_gen = Fd.StockDataGenerator()

    # Generate stock data with custom parameters and a start date
    stock_gen.TuneMyData(StartingPrice=150, Variance=0.02, Runif=6000, start_date='2000-02-01')
    ConsoleModel(stock_gen.GetOHLCData())




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
