import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words, stopwords
from nltk.stem import PorterStemmer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy import sparse  # For sparse matrices to save memory

from NB_Classifier import NaiveBayes

nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)

class MLxEngine:
    def __init__(self):
        self.data = pd.read_csv("data/emails.csv")
        self.vocabulary = {}
        self.set_words = set(words.words())
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
        # Experiment options
        self.use_stemming = False
        self.remove_stopwords = False
        self.vocab_mode = 'english'  # 'english' or 'all'
        self.feature_type = 'bow'    # 'bow' or 'tfidf'
        self.use_bigram = False
        self.min_freq = 2            # Min frequency to filter rare terms
        
        self.X = None  # Sparse matrix
        self.y = None
        self.idf = None

    def _get_tokens(self, email_words):
        # Process tokens for an email
        tokens = []
        for word in email_words:
            lower = word.lower()
            if self.vocab_mode == 'english' and lower not in self.set_words:
                continue
            if self.remove_stopwords and lower in self.stop_words:
                continue
            key = self.stemmer.stem(lower) if self.use_stemming else lower
            tokens.append(key)
        return tokens

    def Build_Vocabulary(self, 
                         use_stemming=False, 
                         remove_stopwords=False,
                         vocab_mode='english',
                         use_bigram=False,
                         min_freq=2):
        # Build vocabulary with options
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.vocab_mode = vocab_mode
        self.use_bigram = use_bigram
        self.min_freq = min_freq
        self.vocabulary = {}
        
        # Count frequencies
        temp_counter = {}
        for i in range(self.data.shape[0]):
            email_words = self.data.iloc[i, 0].split()
            tokens = self._get_tokens(email_words)
            
            # Unigrams
            for token in set(tokens):
                temp_counter[token] = temp_counter.get(token, 0) + 1
            
            # Bigrams
            if use_bigram and len(tokens) >= 2:
                for j in range(len(tokens)-1):
                    bigram = tokens[j] + " " + tokens[j+1]
                    temp_counter[bigram] = temp_counter.get(bigram, 0) + 1
        
        # Filter by min_freq
        idx = 0
        for term, freq in temp_counter.items():
            if freq >= min_freq:
                self.vocabulary[term] = idx
                idx += 1
        
        print(f"Vocabulary size: {len(self.vocabulary)} (min_freq={min_freq})")

    def vocabulary_to_vector(self, feature_type='bow'):
        # Vectorize emails as sparse matrix
        if not self.vocabulary:
            raise ValueError("Build vocabulary first")
        
        self.feature_type = feature_type
        
        rows, cols, data = [], [], []
        self.y = np.zeros(self.data.shape[0])

        for i in tqdm(range(self.data.shape[0])):
            email_words = self.data.iloc[i, 0].split()
            self.y[i] = self.data.iloc[i, 1]
            tokens = self._get_tokens(email_words)
            
            # Unigram counts
            for token in tokens:
                if token in self.vocabulary:
                    rows.append(i)
                    cols.append(self.vocabulary[token])
                    data.append(1)
            
            # Bigram counts
            if self.use_bigram and len(tokens) >= 2:
                for j in range(len(tokens)-1):
                    bigram = tokens[j] + " " + tokens[j+1]
                    if bigram in self.vocabulary:
                        rows.append(i)
                        cols.append(self.vocabulary[bigram])
                        data.append(1)
        
        # Create sparse matrix with float dtype
        self.X = sparse.csr_matrix((data, (rows, cols)), 
                                   shape=(self.data.shape[0], len(self.vocabulary)),
                                   dtype=np.float64)  # Fix dtype for TF-IDF ops

        # Apply TF-IDF if selected
        if feature_type == 'tfidf':
            doc_freq = np.array((self.X > 0).sum(axis=0)).squeeze() + 1
            self.idf = np.log(self.data.shape[0] / doc_freq)
            row_sums = np.array(self.X.sum(axis=1)).squeeze()
            row_sums[row_sums == 0] = 1
            self.X = self.X.multiply(1 / row_sums[:, np.newaxis])
            self.X = self.X.multiply(self.idf)

    def train(self, model):
        if self.X is None or self.y is None:
            raise ValueError("Vectorize data first")
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        model.fit(X_train.toarray(), y_train)  # Convert to dense for fit
        
        y_fit = model.predict(X_train.toarray())
        y_pred = model.predict(X_test.toarray())
        
        fit_accuracy = np.mean(y_fit == y_train)
        accuracy = np.mean(y_pred == y_test)
        print(f"Train Accuracy: {fit_accuracy:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")

    def predict(self, raw_email, model):
        if not self.vocabulary:
            raise ValueError("Build vocabulary first")
        
        email_words = raw_email.split()
        tokens = self._get_tokens(email_words)
        rows, cols, data = [], [], []
        
        # Unigram counts
        for token in tokens:
            if token in self.vocabulary:
                rows.append(0)
                cols.append(self.vocabulary[token])
                data.append(1)
        
        # Bigram counts
        if self.use_bigram and len(tokens) >= 2:
            for j in range(len(tokens)-1):
                bigram = tokens[j] + " " + tokens[j+1]
                if bigram in self.vocabulary:
                    rows.append(0)
                    cols.append(self.vocabulary[bigram])
                    data.append(1)
        
        # Create sparse vector with float dtype
        email_vector = sparse.csr_matrix((data, (rows, cols)), 
                                         shape=(1, len(self.vocabulary)),
                                         dtype=np.float64)  # Fix dtype
        
        # Apply TF-IDF if selected
        if self.feature_type == 'tfidf':
            total = email_vector.sum()
            if total > 0:
                email_vector = email_vector.multiply(1 / total)
            if self.idf is not None:
                email_vector = email_vector.multiply(self.idf)
        
        pred_class = model.predict(email_vector.toarray())[0]
        return "Spam" if pred_class == 1 else "Not Spam"
    



if __name__ == "__main__":
    engine = MLxEngine()
  
    
    

    # BoW
    engine.Build_Vocabulary()
    engine.vocabulary_to_vector('bow')
    model = NaiveBayes()
    engine.train(model)
    prediction = engine.predict("Buy cheap money now!", model)
    print(prediction)

    # stemming + stop word + TF-IDF
    model = NaiveBayes()
    engine.Build_Vocabulary(use_stemming=True, remove_stopwords=True)
    engine.vocabulary_to_vector('tfidf')
    engine.train(model)
    prediction = engine.predict("Buy cheap money now!", model)
    print(prediction)

    # bigram
    model = NaiveBayes()
    engine.Build_Vocabulary(use_stemming=True, remove_stopwords=True, use_bigram=True)
    engine.vocabulary_to_vector('bow')
    engine.train(model)
    prediction = engine.predict("Buy cheap money now!", model)
    print(prediction)
 