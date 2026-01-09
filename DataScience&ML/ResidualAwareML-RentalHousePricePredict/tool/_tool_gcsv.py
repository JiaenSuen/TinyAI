import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

input_csv = 'DATA/data4k.csv'
output_dir = 'DATA/'
features = [ 'size', 'age', 'bedroom', 'CP_value']
label = 'price'
test_ratio = 0.2


df = pd.read_csv(input_csv)

#  pattern 
def extract_pattern(pattern):
    result = {
        'living': 0,
        'bath': 0,
        'balcony': 0,
        'kitchen': 0
    }
    if pd.isna(pattern):
        return result

    if '開放式格局' in pattern:
        return result

    left_part = pattern.split('-')[0] if '-' in pattern else pattern
    matches = re.findall(r'(\d)廳|(\d)衛', left_part)
    for match in matches:
        if match[0]:
            result['living'] = int(match[0])
        if match[1]:
            result['bath']   = int(match[1])

    if '-' in pattern:
        right_part = pattern.split('-')[1]
        if '陽台' in right_part:
            result['balcony'] = 1
        if '廚房' in right_part:
            result['kitchen'] = 1

    return result

pattern_features = df['pattern'].apply(extract_pattern).apply(pd.Series)
df = pd.concat([df, pattern_features], axis=1)

#  environment 
def score_environment(env):
    if pd.isna(env):
        return 0
    return len(env.split(';'))

df['environment_score'] = df['environment'].apply(score_environment)

#Label Encode 
label_encoders = {}
for col in df.select_dtypes(include='object'):
    if col not in [ 'pattern', 'environment', 'title']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le


target_columns = features + [ 'city','house_type','living', 'bath', 'balcony', 'kitchen', 'environment_score', label]
df = df[target_columns].dropna()


X = df.drop(columns=[label])
y = df[label]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=42
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df  = pd.concat([X_test, y_test], axis=1)


train_df.to_csv(output_dir + 'train.csv', index=False, encoding='utf-8-sig')
test_df.to_csv(output_dir + 'test.csv', index=False, encoding='utf-8-sig')
