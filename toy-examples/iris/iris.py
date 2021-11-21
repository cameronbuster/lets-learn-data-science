from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler


hyperparameters = {'classifier__subsample': [1.0, 2.5, 5.0, 10.0],
                   'classifier__eta': [0.001, 0.01, 0.3, 0.6],
                   'classifier__max_delta_step': [0, 1, 5, 10],
                   'classifier__num_round': [10, 50, 100, 500]
}


def feature_engineering(X):
    X['e'] = X['a']*X['b']
    return X


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target #numpy array
    X_df = pd.DataFrame(data=X)
    # how do we get y (numpy array) -> pandas dataframe?
    y_df = pd.DataFrame(data=y)

    X_df = X_df.rename(columns={0: 'a', 1: 'b', 2: 'c', 3: 'd'})
    y_df = y_df.rename(columns={0: 'target'})
    df = pd.concat([X_df.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2)
    print(X_train)

    pipeline = Pipeline(
        [
            ('feature_engineering', feature_engineering()),
            ('variance_threshold', VarianceThreshold()),
            ('min_max', MinMaxScaler()),
            ('classifier', XGBClassifier())
        ]
    )
    grid_search = GridSearchCV(
        pipeline,
        param_grid=hyperparameters,
        n_jobs=-1,
        cv=5,
        scoring='roc_auc',
        return_train_score=True,
        verbose=3
    )
    pipe = grid_search.fit(X_train, y_train)
    print(pipe)
    y_preds = pipe.predict(X_test)
    print(y_preds)
    print(classification_report(y_test, y_preds))
    print(confusion_matrix(y_test, y_preds))
    df_cm = pd.DataFrame(confusion_matrix(y_test, y_preds))
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


if __name__=="__main__":
    pd.options.display.max_rows = 999
    main()