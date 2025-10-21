# 2024 record -- Project Code : 04 --
# Model :  Decision Tree + Parameter Search


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score
from sklearn import tree
import graphviz

 
df = pd.read_csv('data/weather_data_year.csv')
if 'Rain' not in df.columns:
    df['Rain'] = (df['Precipitation'] > 0).astype(int)

encoder = LabelEncoder()
df['WindDirection'] = encoder.fit_transform(df['WindDirection'])

features = ['AirPressure', 'AirTemperature', 'RelativeHumidity', 
            'WindSpeed', 'WindDirection', 'SunshineDuration','Month', 'Hour']
X = df[features]
y = df['Rain']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
X_train_main, X_val, y_train_main, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)






# Define parameter range
max_depth_values = [2,3,5,8,10,12,15,20]  # Maximum depth
min_samples_split_values = [2, 5 ,10]  # Minimum number of samples for splitting a node
min_samples_leaf_values = [1, 2,5 ]    # Minimum number of samples per leaf node

best_model = None
best_auc = 0
best_params = {}

# Manually test different parameter combinations and use cross-validation
for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        for min_samples_leaf in min_samples_leaf_values:
            model = DecisionTreeClassifier(max_depth=max_depth, 
                                          min_samples_split=min_samples_split, 
                                          min_samples_leaf=min_samples_leaf, 
                                          random_state=42,
                                          class_weight='balanced')
            model.fit(X_train_main, y_train_main)
            y_val_proba = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val, y_val_proba)
            auc_score = auc(fpr, tpr)

           # Update the best model
            if auc_score > best_auc:
                best_auc = auc_score
                best_model = model
                best_params = {'max_depth': max_depth, 
                               'min_samples_split': min_samples_split, 
                               'min_samples_leaf': min_samples_leaf}

# Output the best parameters and AUC
print(f"Optimal parameters : {best_params}")
print(f"Best validation set AUC : {best_auc:.2f}")

# Use the best model to make predictions on the test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Evaluation results
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision tree model test set accuracy : {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))



# Plotting the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("Record/DecisionTree/DecisionTree_CM.jpg")
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
plt.savefig("Record/DecisionTree/DecisionTree_ROC.jpg")
plt.show()

# Visualizing Decision Trees
dot_data = tree.export_graphviz(best_model, out_file=None, 
                                feature_names=features,  
                                class_names=["No Rain", "Rain"],  
                                filled=True, rounded=True,  
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph.render("Record/DecisionTree/Dedecision_tree")  
graph.view()
