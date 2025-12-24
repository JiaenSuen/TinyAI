# 2024 record -- Project Code : 06 --
# Model :  XGBoosting + SMOTE


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV

# Data
df = pd.read_csv('data/weather_data_year.csv')
 
if 'Rain' not in df.columns:
    df['Rain'] = (df['Precipitation'] > 0).astype(int)

encoder = LabelEncoder()
df['WindDirection'] = encoder.fit_transform(df['WindDirection'])

features = ['AirPressure', 'AirTemperature', 'RelativeHumidity', 
            'WindSpeed', 'WindDirection', 'SunshineDuration', 'Month', 'Hour']
X = df[features]
y = df['Rain']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)



# SMOTE : For Oversampling
smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.8)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Calculate scale_pos_weight
scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

# Train the XGBoost model
model = XGBClassifier(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=2,
    max_delta_step=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# Reduce the GridSearchCV search range
param_grid = {
    'max_depth': [6, 7, 8],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [150, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 0.9]
}
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring=make_scorer(f1_score, average='binary'),
    cv=3,
    verbose=1
)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

# Predict test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability prediction

# Get probability prediction
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model test set accuracy : {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plotting the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Record/XGBoosting/XGBoosting_CM.jpg")
plt.show()

# Plotting ROC curve and calculating AUC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.savefig("Record/XGBoosting/XGBoosting_ROC.jpg")
plt.show()
