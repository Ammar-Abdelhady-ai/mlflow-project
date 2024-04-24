import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os, mlflow, argparse
from PIL import Image
import numpy as np
from sklearn_features.transformers import DataFrameSelector
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, accuracy_score, auc
import joblib


cwd = os.getcwd()
df = pd.read_csv(os.path.join(cwd, "dataset.csv"))

df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)
df.drop(index=df[df["Age"] > 80].index.to_list(), axis=0, inplace=True)

X = df.drop("Exited", axis=1)
y = df["Exited"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, 
                                                    shuffle=True, stratify=y)

num_cols = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
cat_cols = ['Geography', 'Gender']

ready_cols = list(set(X_train.columns.to_list()) - set(cat_cols) - set(num_cols))



# Pipeline

num_pipeline = Pipeline(steps=[
        ("selector", DataFrameSelector(num_cols)), 
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()), 
    ], verbose=True
)

cat_pipeline = Pipeline(steps=[
        ("selector", DataFrameSelector(cat_cols)), 
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoded", OneHotEncoder(drop="first", sparse_output=False)),
    ], verbose=True
)

ready_pipeline = Pipeline(steps=[
        ("selector", DataFrameSelector(ready_cols)),  
        ("imputer", KNNImputer(n_neighbors=5)),
    ], verbose=True
)

# Concatenates results of multiple transformer objects.
all_pipeline = FeatureUnion(transformer_list=[
        ("numerical", num_pipeline), 
        ("categorical", cat_pipeline),
        ("ready", ready_pipeline),
    ], n_jobs=-1, verbose=True
)


X_train_final = all_pipeline.fit_transform(X_train)
X_test_final = all_pipeline.transform(X_test)


# Count number of occurrences of each value in array fron zero to large...

np.bincount(y)

# To get ratio
np.bincount(y) / len(y)


# to reverse ratio to add it as weights for model
val_count = 1 - (np.bincount(y_train) / len(y_train))
val_count = val_count / np.sum(val_count) # To nurmalize


dict_weight = {}

for i in range(y.nunique()):
    dict_weight[i] = val_count[i]
dict_weight

smote = SMOTE(sampling_strategy=0.8)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train_final, y_train)

# RandomForestClassifier(class_weight=dict_weight)
with open("metrics.txt", "w") as f:
    pass


def train_model(X_train, y_train, plot_name, splitter: str, max_depth: int, class_weight=None):
    clf = DecisionTreeClassifier(splitter=splitter, max_depth=max_depth, random_state=45, criterion="gini", class_weight=class_weight)
    clf_name = clf.__class__.__name__

    mlflow.set_experiment(experiment_name="churn-detection")
    with mlflow.start_run() as run:
        mlflow.set_tag("clf", clf_name)
        clf.fit(X_train, y_train)

        y_test_predict = clf.predict(X_test_final)

        score_test = f1_score(y_test, y_test_predict)
        acc_test = accuracy_score(y_test, y_test_predict)

        mlflow.log_params({"splitter": splitter, "max_depth": max_depth})
        mlflow.log_metrics({"Accuracy": acc_test, "F1_Score": score_test})
        mlflow.sklearn.log_model(clf, f"{clf_name}/{plot_name}")


        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)  
        sns.heatmap(confusion_matrix(y_test, y_test_predict), annot=True, fmt=".2f", cmap="Blues", cbar=False)
        
        plt.title(f'{plot_name}')
        plt.xticks(ticks=np.arange(2) + 0.5, labels=[False, True])
        plt.yticks(ticks=np.arange(2) + 0.5, labels=[False, True])
        
        conf_fig = plt.gcf()
        mlflow.log_figure(figure=conf_fig, artifact_file=f"{plot_name}_confusion_matrix.png")
        plt.close()
        # Plotting the ROC curve
        fpr, tpr, thresh = roc_curve(y_test, y_test_predict)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) {plot_name}')
        plt.legend(loc="lower right")
        conf_fig = plt.gcf()

        mlflow.log_figure(figure=conf_fig, artifact_file=f"{plot_name}_roc_curve.png")


        plt.close()


    return True

def main(splitter: str, max_depth: int):
    train_model(X_train=X_train_final, y_train=y_train, plot_name="without-imbalance", 
                splitter=splitter, max_depth=max_depth, class_weight=None)

    train_model(X_train=X_train_final, y_train=y_train, plot_name="with-class-weight", 
                splitter=splitter,  max_depth=max_depth, class_weight=dict_weight)

    train_model(X_train=X_train_resampled, y_train=y_train_resampled, plot_name="with-SMOT", 
                splitter=splitter, max_depth=max_depth, class_weight=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splitter", "-s", type=str, default="best")
    parser.add_argument("--max_depth", "-d", type=int, default=15)
    args = parser.parse_args()


    main(splitter=args.splitter, max_depth=args.max_depth)