import pandas as pd
import os

from sklearn.model_selection import train_test_split

dataset = "C:/Playground/Python/lets-learn-data-science/not-so-toy-examples/imbalanced-classification/dataset/creditcard.csv"

def main():
    pwd = os.getcwd()
    os.chdir(os.path.dirname(dataset))
    df = pd.read_csv(os.path.basename(dataset))
    os.chdir(pwd)
    print(df.head)
    print(df.dtypes)

    X = df.loc[:, 'Time':'Amount']
    y = df.Class

    print(y.head)
    print(X.head)

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=1991)

if __name__=="__main__":
    main()