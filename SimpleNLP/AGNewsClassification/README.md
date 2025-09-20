## **AGNews Classification with PyTorch**
### August 2025
 
This project is about classifying news articles into four categories: World, Sports, Business, and Sci/Tech. It uses deep learning with PyTorch. We use a simple LSTM model and pre-trained GloVe word embeddings to make it easy for beginners. The data comes from AG News dataset.
The project has two main files:

TrainingModel.ipynb: This is a Jupyter Notebook for training and testing the model.
agnews_model.py: This is a Python module with helper functions and the model class.

 
## Project Overview
Goal: Train a model to read news text (title + description) and predict the category.
Tools Used:

PyTorch: For building and training the neural network.
NLTK: For tokenizing text (splitting into words).
GloVe: Pre-trained word vectors to represent words as numbers.
Pandas: For loading CSV data.
 

Train: 120,000 news articles.
Test: 7,600 news articles.
Categories: 0=World, 1=Sports, 2=Business, 3=Sci/Tech.

## Functions and Classes

### AGNewsDataset
Custom dataset class to prepare text and labels for PyTorch. It helps the model process news articles by converting text into numerical tokens and padding them to a fixed length, making data consistent for training and testing.

### build_vocab
Creates a vocabulary dictionary from text data. It’s needed to map words to unique IDs, allowing the model to understand text as numbers for processing.

### load_glove_vectors
Loads pre-trained GloVe word embeddings from a file. This provides ready-made word representations, saving time and improving model performance with meaningful vectors.

### create_embedding_matrix
Builds a matrix of word embeddings for the vocabulary. It’s essential to give the model a starting point with pre-trained vectors, ensuring words have numerical meaning.


###  predict_single
Predicts the category of a single news article. It’s useful for applying the trained model to new text, making it practical for real-world use.

## Model Introduction
The TextClassifier model is a simple deep learning architecture designed for AG News classification, balancing ease of understanding and performance for beginners. It uses an Embedding layer to convert words into 100-dimensional GloVe vectors, which capture word meanings efficiently. An LSTM layer processes the sequence of word vectors to understand context and relationships in the text, chosen for its ability to handle sequential data like sentences. Finally, a Linear layer maps the LSTM’s output to four classes (World, Sports, Business, Sci/Tech). This design is straightforward yet effective, leveraging pre-trained embeddings to reduce training time and an LSTM to capture text patterns, making it suitable for text classification tasks.