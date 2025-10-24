# from Vocabulary map into X,Y dataset
# [aardkvark, ..., buy, ..., money, ..., zuzu]
# [0, ...,          8,  ..., 10   , ..., 0]


import pandas as pd
import numpy  as np
from tqdm import tqdm
import ast

data = pd.read_csv("data/emails.csv")
file = open("data/vocabulary.txt",'r')
contents = file.read()
vocabulary = ast.literal_eval(contents)

X = np.zeros((data.shape[0],len(vocabulary)))
y = np.zeros((data.shape[0]))

for i in tqdm(range(data.shape[0])):
    email = data.iloc[i,0].split()

    for email_word in email:
        if email_word.lower() in vocabulary:
            X[i,vocabulary[email_word]] += 1
            y[i] = data.iloc[i,1]



np.save("data/X.npy",X)
np.save("data/y.npy",y)