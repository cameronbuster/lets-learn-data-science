from sklearn.datasets import load_boston


def main():
    X, y = load_boston(return_X_y=True)


if __name__=="__main__":
    main()