from PiecewiseRegressionML import SegmentedModel

import pandas as pd
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


 
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

 
features = [ 'size','age','house_type','bedroom','living','bath','balcony','kitchen','environment_score' ]
X_train = train[features]
y_train = train['price']
X_test  = test[features]
y_test  = test['price']


list_of_combination = [ 
    {"SmallModel" : XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42), "LargeModel" :XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42) },
    {"SmallModel" : GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42), "LargeModel" :GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42) },
]


for combination in list_of_combination:

    print("================================================================================================")
    print(f"Small Model : {combination['SmallModel'].__class__.__name__}  Large Model : {combination['LargeModel'].__class__.__name__} ")
    S_model = combination["SmallModel"]
    L_model = combination["LargeModel"]

    seg_model = SegmentedModel(
        split_col="size",  
        split_value=None, 
        model_small=S_model,
        model_large=L_model
    )


    seg_model.fit(X_train, y_train)
    y_pred = seg_model.predict(X_test)
    print(f"Segment ML R² Score Train : {r2_score(y_train, seg_model.predict(X_train)):.4f}  Test : {r2_score(y_test, y_pred):.4f}\n")