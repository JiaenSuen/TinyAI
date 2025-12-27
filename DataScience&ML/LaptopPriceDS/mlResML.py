import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)


from modules.FeatureCoder import FeatureEncoderV2
import matplotlib.pyplot as plt

from modules.ResdiualML import Residual_Stack

df = pd.read_csv("laptop_price.csv", encoding="latin1")

encoder = FeatureEncoderV2(max_tfidf_features=100)
df_feat = encoder.fit_transform(df)
encoder.save("pkls/feature_encoder_v2.pkl")

X = df_feat.drop("Price_euros", axis=1)
y = df_feat["Price_euros"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ---- Models ---
model_xgb = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    base_score=y_train.mean()
)
model_ada = AdaBoostRegressor(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
)

model_rdf = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)

model_ext = ExtraTreesRegressor(
    n_estimators=100,
    random_state=42,
)

# ---- Models ---



model =  Residual_Stack(
    base_model=model_xgb,
    residual_model=model_rdf
)


model.fit(X_train, y_train)
pred = model.predict(X_test)
print("R2:", r2_score(y_test, pred))


 
# R2: 0.8436618034951031
