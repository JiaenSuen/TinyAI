import pandas as pd
import numpy  as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
    

# download from kaggle TMDB dataset
df_movies  = pd.read_csv("tmdb/tmdb_5000_movies.csv ")
df_credits = pd.read_csv("tmdb/tmdb_5000_credits.csv")


# TF*IDF Algorithm : Term Frequency * Inverse Document Frequency
tfidf = TfidfVectorizer(stop_words="english")
df_movies['overview'] = df_movies['overview'].fillna("")
tfidf_matrix = tfidf.fit_transform(df_movies['overview'])

# Vector Space Model 
cosine_similarity =  linear_kernel(tfidf_matrix,tfidf_matrix)

indices = pd.Series(df_movies.index,index=df_movies['original_title']).drop_duplicates()
 
def get_recommendations(title,cosine_sim = cosine_similarity):
    idx = indices[title]
    sim_scores =  enumerate(cosine_sim[idx]) 
    sim_scores = sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df_movies['original_title'].iloc[movie_indices]
    
recommendation_movies = get_recommendations("The Dark Knight Rises")
print(recommendation_movies)