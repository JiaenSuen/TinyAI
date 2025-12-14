# Low Level Chat Bot with PyTorch
https://www.kaggle.com/datasets/niraliivaghani/chatbot-dataset  
Just a little practice from Kaggle Dataset and simulate a low level chat bot.

## Library / Module
### NLTK
NLTK (Natural Language Toolkit) is a popular Python library for natural language processing, provides tools for tokenization, linguistic analysis, and lexical resources, widely used in NLP research and applications.  
punkt  : Used for sentence splitting and tokenization.  
punkt_tab : Supporting tables for punkt, improving sentence boundary detection.  
wordnet: An English lexical database used for synonyms and lemmatization.  

### PyTorch
https://pytorch.org/get-started/locally/  
PyTorch is an open-source Python machine learning library. In this little project, you can use it to build and manage the Artificial Neural Networks & Deep Learning Modules.


## Natural Language Processing , NLP
### Tokenize
Splits text into meaningful units such as words or sentences, making the text processable by NLP models.  
### Lemmatize
Converts words to their base form (e.g., running → run), helping models understand true word meaning.
### Bag of Words
Represents text as word frequency vectors, ignoring grammar and word order, commonly used in text classification.

## Project System FlowChart :
Human sentence input → Tokenize → Lemmatize  → Bag of Words → Transform to Tensor → ANN ( Multilayer Perceptron ) → Intent Prediction → Random pick a respone from document.