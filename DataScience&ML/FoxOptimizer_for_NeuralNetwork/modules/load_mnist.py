import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 

def load_mnist_inCSV(path_of_dataset ,test_show=False):
    data = pd.read_csv(path_of_dataset)
    if test_show : print(data.head(5))
 
    

    data=np.array(data)
    m, n= data.shape
    if test_show : print(f"Shape of data : {data.shape}")
 
    np.random.shuffle(data)

    # normalize  
    data=data[0:m].T  # Transpose it make each column is a example, coulda make it easier.
    Y=data[0].astype(int)      
    X=data[1:n]
    X = X / 255.
    _,m_x = X.shape

    if test_show : print(f"X shape : {Y.shape}")
    if test_show : print(f"Y shape : {Y.shape}")

    return X,Y