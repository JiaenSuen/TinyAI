## **Pizza Reviws ReadingBot with Langchain Context Engineering**

This a little test side Q&A Project for learning about langchain , <br>
It's a tiny case make LLM model to read the reviews which helpful for user's quesions and give user better respone.<br> So , how is it work?  <br>

There's a CSV file containing Title, Rating and Review needs to be prepared in the project. Each Review data needs to be packaged into a Document for vector database. I use Langchain to connect Chroma vector database and LLM by using a embedding model and retriever to search. <br>

Retriever will use cosine similarity to search and return k reviews which most similar to user's question. In the end , it can add those reviews into prompt for LLM, so we can get respone better quality and authentic responses.

## Framework

LangChain is a development framework for building and interacting with Large Language Models. Providing chained tasks, prompt management, data retrieval, and other features. Enable language models to understand and process information in specific fields more accurately.

Chroma is a vector database for storing and retrieving high-dimensional vector data, such as text, images, or other feature vectors.

## Model

Ollama is a tool and platform for local large-scale language model operations, providing a simple interface to call models, such as the LLaMA series

LLaMA 3 is Meta's third-generation Large Language Model (LLM), offering powerful natural language understanding and generation capabilities.

## Result Example

Question : Can you recommend some pizza dishes with their name ?
Bot Answer : 

Based on the reviews, here are some pizza dish recommendations:

* "Avant Garde" menu section features innovative flavor combinations, such as the pear, gorgonzola, walnut, and honey pizza that was a "revelation of sweet, salty, and savory harmony".
* "White Album" pizza with ricotta, mozzarella, garlic, and spinach is phenomenal.
* Fig and prosciutto pizza is excellent.
* Sausage pizza pairs well with the house red wine.

Additionally, they have traditional options, creative specialties, and customizable choices. And don't forget to try their dessert pizzas, such as Nutella and banana!
