import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from geopy.geocoders import Nominatim 

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
import os

df = pd.read_csv('data/us_cities_ml_dataset.csv', sep=',')


country = 'United States'
city_names = df['location']

longitude =[]
latitude =[]
geolocator = Nominatim(user_agent="Trips")

for c in city_names.values:
    location = geolocator.geocode(c+','+ country)
    latitude.append(location.latitude)
    longitude.append(location.longitude)


df ['longitude'] = longitude
df ['latitude']  = latitude

l2 = df.iloc[:,1:3:1]


kmeans = KMeans(5)
kmeans.fit(l2)

identified_clusters = kmeans.fit_predict(l2)
identified_clusters = list(identified_clusters)
df['loc_clusters']  = identified_clusters

os.makedirs("output",exist_ok=True)
df.to_csv('output/clusted_df.csv')



while True:
    input_city = input("Enter a city name: ")#New York
    if input_city == 'q' : break
    print("input : ",input_city,'\n') 
    cluster = df.loc[df['location'] == input_city, 'loc_clusters']
    cluster = cluster.iloc[0]
    cluster
    cities = df.loc[df['loc_clusters'] == cluster, 'location']
    cities
    for c in range(len(cities)): 
        if cities.iloc[c] == input_city:
            continue
        else:
            print(cities.iloc[c])