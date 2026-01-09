from PiecewiseRegressionML import SegmentedModel
from ResdiualML import Residual_Stack

import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import r2_score


 
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

 
features = [ 'size','age','house_type','bedroom','living','bath','balcony','kitchen','environment_score' ]
X_train = train[features]
y_train = train['price']
X_test  = test[features]
y_test  = test['price']



XGB = XGBRegressor(n_estimators=50, learning_rate=0.05, random_state=42)
ADA = AdaBoostRegressor(n_estimators=50, learning_rate=0.05, random_state=42)
S_RAmodel_XGB_ADA  = Residual_Stack(base_model=XGB, residual_model=ADA)
L_RAmodel_XGB_ADA  = Residual_Stack(base_model=XGB, residual_model=ADA)

seg_model = SegmentedModel(
    split_col="size",  
    split_value=None, 
    model_small=S_RAmodel_XGB_ADA,
    model_large=L_RAmodel_XGB_ADA
)

seg_model.fit(X_train, y_train)
y_pred = seg_model.predict(X_test)
score = seg_model.score(X_test, y_test)
print(f"Segment Res[XGB,GB] R² Score Train : {r2_score(y_train, seg_model.predict(X_train)):.4f}  Test : {r2_score(y_test, y_pred):.4f}")
