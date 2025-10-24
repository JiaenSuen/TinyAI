import pandas as pd
import numpy  as np
import nltk
from nltk.corpus import words
from tqdm import tqdm

vocabulary = {}
data = pd.read_csv("data/emails.csv")
nltk.download('words')
set_words = set(words.words())

def build_vocabulary(curr_email):
    idx = len(vocabulary)

    for word in curr_email:
        if word.lower() not in vocabulary and word.lower() in set_words:
            vocabulary[word] = idx  
            idx +=1 


if __name__ == '__main__':
    for i in tqdm(range(data.shape[0])):
        curr_email = data.iloc[i,0].split()
        build_vocabulary(curr_email)
        
    print(f"the Length of vocabulary is {len(vocabulary)}")
    file = open("data/vocabulary.txt","w")
    file.write(str(vocabulary))
    file.close()
