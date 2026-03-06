import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import Perceptron , PassiveAggressiveClassifier
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,XGBRegressor
from ResidualML_ClassificationVersion import Residual_Stack_Classifier
from itertools import combinations

# dataset : https://www.kaggle.com/datasets/gauravkumar2525/kepler-exoplanet-dataset



RND_SEED = 20260305


def clean_nan(df):

    df = df.copy(deep=True)

    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].fillna("Unknown")

    return df
 
def feature_processing(df, target_col=None):

    df = df.copy(deep=True)

    # delete ID
    df = df.iloc[:, 1:]

    if target_col is None:
        target_col = df.columns[-1]

    encoders = {}

    str_cols = df.select_dtypes(include=["object", "string"]).columns

    for col in str_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # split X Y
    X = df.drop(columns=[target_col])

    le_y = LabelEncoder()
    Y = le_y.fit_transform(df[target_col])
    encoders["target"] = le_y

    return X, Y, encoders
 

def auto_feature_engineering_with_selection(df, target_col=None, max_interact=2):
    df = df.copy(deep=True)
    
    if target_col is None:
        target_col = df.columns[-1]

 
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    
    cat_encoders = {}
    for col in cat_cols:
        if df[col].nunique() <= 10:  # A few categories do One-Hot
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else: # Use LabelEncoder for high cardinality categories
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            cat_encoders[col] = le

    # Interactive features
    new_features = {}
    for comb_len in range(2, max_interact+1):
        for comb in combinations(num_cols, comb_len):
            name = "_x_".join(comb)
            df[name] = df[list(comb)].prod(axis=1)  
            new_features[name] = comb

    # Use Perceptron to select the most effective features
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    # Perceptron Rate All Feature
    scores = {}
    for col in X.columns:
        model = Perceptron(random_state=RND_SEED, max_iter=5000)
        model.fit(X[[col]], y)
        score = model.score(X[[col]], y)
        scores[col] = score

    threshold = 0.5
    selected_features = [k for k,v in scores.items() if v >= threshold]
    X_selected = X[selected_features]

    print("Selected Features:", selected_features)
    return X_selected, y, {"cat_encoders": cat_encoders, "new_features": new_features, "selected_features": selected_features}

dataRaw = pd.read_csv("exoplanets_data.csv")
dataRaw = clean_nan(dataRaw)



activate_fe = input("Performing automated feature engineering (Yes : 1, No : 0) ? : ")
if activate_fe=="1":
    trainX, trainY, fe_info = auto_feature_engineering_with_selection(
        dataRaw, target_col="koi_disposition"
    )
else:
    trainX, trainY, encoders = feature_processing(
        dataRaw,target_col="koi_disposition"
    )




trainX, testX, trainY, testY = train_test_split(
    trainX, trainY, train_size=0.8, shuffle=True,random_state=RND_SEED,
)



import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import List 

def run_models(models, trainX, trainY, testX, testY, classes_list: List[str], subdir="record", visualize=False, save=True):
    results = {}

    for name, model in models.items():
        model.fit(trainX, trainY)
        score = model.score(testX, testY)
        results[name] = score
        print(f"{name}: {score:.4f}")

        # Additional evaluation
        y_pred = model.predict(testX)
        acc = accuracy_score(testY, y_pred)
        report = classification_report(testY, y_pred, target_names=classes_list, zero_division=0)
        cm = confusion_matrix(testY, y_pred)

        if save:
            REPORT_FILENAME = os.path.join(subdir, f"{name}_report.txt")
            with open(REPORT_FILENAME, "w", encoding='utf-8') as f:
                f.write(f"Accuracy: {acc:.4f}\n\n")
                f.write(report)
            #print(f"Result Save : {REPORT_FILENAME}")

        if visualize or save:
            fig = plt.figure(figsize=(14, 6))
            # Left: Confusion Matrix
            ax1 = fig.add_subplot(1, 2, 1)
            sns.heatmap(cm, annot=True, fmt="d",
                        xticklabels=classes_list, yticklabels=classes_list,
                        cmap="Blues", cbar=False, ax=ax1)
            ax1.set_xlabel("Predicted label")
            ax1.set_ylabel("True label")
            ax1.set_title(f"Confusion Matrix - {name}")

            # Right: Classification Report
            ax2 = fig.add_subplot(1, 2, 2)
            ax2.axis('off')
            ax2.text(0, 1.0, f"Accuracy : {acc:.4f}",
                     fontsize=14, fontweight='bold', va='top')
            ax2.text(0, 0.9, "Classification Report:",
                     fontsize=13, fontweight='bold')
            ax2.text(0, 0.8, report,
                     fontfamily='monospace', fontsize=11.5,
                     va='top', linespacing=1.4)
            plt.suptitle(f"Model Evaluation - {name}", fontsize=16, y=1.02)
            plt.tight_layout()

            if save:
                COMBINED_FILENAME = os.path.join(subdir, f"{name}_cm_report.png")
                plt.savefig(COMBINED_FILENAME, dpi=150, bbox_inches='tight')
                #print(f"Combined CM + Report saved: {COMBINED_FILENAME}")

            if visualize:
                plt.show()

            plt.close()

    return results


models = {
    "Perceptron": Perceptron(random_state=RND_SEED),
    "Passive Aggressive" : PassiveAggressiveClassifier(random_state=RND_SEED),
    "SVM" : SVC(random_state=RND_SEED),
    "Decision Tree": DecisionTreeClassifier(random_state=RND_SEED),
    "XGBoost": XGBClassifier(random_state=RND_SEED),
    "AdaBoost": AdaBoostClassifier(random_state=RND_SEED , algorithm="SAMME"),
    "RandomForest": RandomForestClassifier(random_state=RND_SEED),
    "HistGradientBoost" : HistGradientBoostingClassifier(random_state=RND_SEED),
    "Residual-Aware-Classifier-XG" :  Residual_Stack_Classifier(
                                        base_model=XGBClassifier(n_estimators=50, random_state=RND_SEED),
                                        residual_model= XGBRegressor(n_estimators=50, random_state=RND_SEED)
                                    ),
}

results = run_models(models, trainX, trainY, testX, testY ,classes_list=["False positives","Candidates","Confirmed"])