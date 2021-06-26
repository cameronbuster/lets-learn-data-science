from sklearn.datasets import load_diabetes


def main():
    X, y = load_diabetes(return_X_y=True)


if __name__=="__main__":
    main()