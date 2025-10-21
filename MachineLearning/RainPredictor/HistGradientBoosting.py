# 2024 record -- Project Code : 07 --
# Model :  Hist Gradient Boosting

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier   

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


# Add SMOTE will make HGB model becoming less adept at handling imbalance



# Train the HistGradientBoosting model
model = HistGradientBoostingClassifier(max_iter=10000, max_depth=100,max_leaf_nodes=100, random_state=42,class_weight='balanced')
model.fit(X_train, y_train)

# Predict test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get probability prediction
    
# Evaluation results
accuracy = accuracy_score(y_test, y_pred)
print(f"HistGradientBoosting Model test set accuracy : {accuracy:.2f}")
print("\nClassification Report :\n", classification_report(y_test, y_pred))

# Plotting the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Record/HistGradientBoosting/HistGradientBoosting_CM.jpg")
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
plt.savefig("Record/HistGradientBoosting/HistGradientBoosting_ROC.jpg")
plt.show()
