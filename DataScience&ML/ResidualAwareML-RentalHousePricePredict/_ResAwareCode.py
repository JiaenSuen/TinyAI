from ResdiualML import Residual_Stack

import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score

 
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

 
features = [ 'size','age','house_type','bedroom','living','bath','balcony','kitchen','environment_score' ]
X_train = train[features]
y_train = train['price']
X_test  = test[features]
y_test  = test['price']

list_of_combination = [ 
    {"BaseModel" : XGBRegressor(n_estimators=150, learning_rate=0.05, random_state=42), "ResdiualModel" :AdaBoostRegressor(n_estimators=150, learning_rate=0.05, random_state=42) },
    {"BaseModel" : XGBRegressor(n_estimators=150, learning_rate=0.05, random_state=42), "ResdiualModel" :GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42) },

]
 
for combination in list_of_combination:

    print("==============================================================================")
    print(f"Base Model : {combination['BaseModel'].__class__.__name__}  Residual Model : {combination['ResdiualModel'].__class__.__name__} ")
    base_model = combination["BaseModel"]
    residual_model = combination["ResdiualModel"] 
    RAmodel  = Residual_Stack(base_model=base_model, residual_model=residual_model)

    RAmodel.fit(X_train, y_train)
    y_pred = RAmodel.predict(X_test)
    print(f"Residual Aware ML R² Score Train : {r2_score(y_train, RAmodel.predict(X_train)):.4f}  Test : {r2_score(y_test, y_pred):.4f}\n\n")
