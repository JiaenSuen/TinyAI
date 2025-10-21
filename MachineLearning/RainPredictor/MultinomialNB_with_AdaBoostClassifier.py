# 2024 record -- Project Code : 02 --
# Model :  Naive Bayes + AdaBoost 

# After using Naive Bayes as first model to apply on this weather predict problem
# And I found that the 2 classification Data is " Unbalanced " 
# So, after a while. I looking for solution on internet 
# Then, I proposed two solution for this case
# One is  "Adaptive Boosting Classifier" in MultinomialNB_with_AdaBoostClassifier.py
# Second one is "Synthetic Minority Oversampling Technique" in  SVM_SMOTE.py


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier

 
df = pd.read_csv('data/weather_data_year.csv')
if 'Rain' not in df.columns:
    df['Rain'] = (df['Precipitation'] > 0).astype(int)
encoder = LabelEncoder()
df['WindDirection'] = encoder.fit_transform(df['WindDirection'])
features = ['AirPressure', 'AirTemperature', 'RelativeHumidity', 
            'WindSpeed', 'WindDirection', 'SunshineDuration', 'Month', 'Hour']
X = df[features]
y = df['Rain']

# With MinMaxScaler : Ensure that the eigenvalues ​​are non-negative
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Calculate category weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
    
# Generate sample weights for each sample
sample_weights = np.array([class_weight_dict[cls] for cls in y_train])

# Use MultinomialNB as the base model
nb_model = MultinomialNB()

# Use AdaBoost for boosting
adaboost_model = AdaBoostClassifier(estimator=nb_model, n_estimators=100, random_state=42)

# Consider sample weights when training the model
adaboost_model.fit(X_train, y_train, sample_weight=sample_weights)

# Predict test set
y_pred = adaboost_model.predict(X_test)
y_pred_proba = adaboost_model.predict_proba(X_test)[:, 1]

# Evaluation results
accuracy = accuracy_score(y_test, y_pred)
print(f"AdaBoost + Multinomial Naive Bayes Test set accuracy : {accuracy:.2f}")
print("\nClassification Report :\n", classification_report(y_test, y_pred))

 
# Plotting the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Record/NB_AdaBoost/NB&Ada_CM.jpg")
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
plt.savefig("Record/NB_AdaBoost/NB&Ada_ROC.jpg")
plt.show()
