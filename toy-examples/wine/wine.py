from sklearn.datasets import load_wine


def main():
    X, y = load_wine(return_X_y=True)


if __name__=="__main__":
    main()