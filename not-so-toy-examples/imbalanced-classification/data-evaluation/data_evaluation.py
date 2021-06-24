import pandas as pd

def main():
    df = pd.read_csv("../dataset/creditcard.csv")
    print(df.head)
    print(df.dtypes)

if __name__=="__main__":
    main()