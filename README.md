<div align="center">

<img src="arctic-fox.jpg" alt="TinyAI Arctic Fox" width="520">

# TinyAI

### Compact AI Experiments Across Machine Learning, NLP, Reinforcement Learning, and More

<p>
  A continuously growing collection of small-scale implementations, technical studies, and proof-of-concept projects.
</p>

<p>
  <a href="#featured-projects">Featured Projects</a> •
  <a href="#project-collections">Project Collections</a> •
  <a href="#repository-guide">Repository Guide</a>
</p>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square\&logo=pytorch\&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Focus-AI%20Experiments-6C63FF?style=flat-square)
![Status](https://img.shields.io/badge/Status-Continuously%20Growing-2EA44F?style=flat-square)

</div>

---

## Overview

**TinyAI** is a curated collection of compact artificial intelligence and machine learning projects developed for learning, reproduction, technical exploration, and rapid hypothesis testing.

The repository covers multiple AI domains, including classical machine learning, natural language processing, reinforcement learning, recommender systems, time-series forecasting, graph neural networks, large-language-model applications, and conversational systems.

Projects in this repository are intentionally small or medium in scope. Larger research projects, publication-related work, and complete systems are generally maintained as standalone repositories.

### Repository Goals

* Build practical understanding through implementation
* Reproduce and study established machine learning methods
* Explore models across different AI domains
* Document technical observations and lessons learned
* Maintain a structured archive of experiments
* Provide reusable references for future research and engineering work

---

<a id="featured-projects"></a>

## Featured Projects

Selected projects that represent the range of topics covered in TinyAI.

| Project                                                                                           | Domain                      | Type            | Summary                                                                                 | Status   |
| ------------------------------------------------------------------------------------------------- | --------------------------- | --------------- | --------------------------------------------------------------------------------------- | -------- |
| [Residual-Aware Rental Price Prediction](DataScience&ML/ResidualAwareML-RentalHousePricePredict/) | Machine Learning            | Publication     | Rental price prediction using a residual-aware learning module; presented at TANet 2025 | Complete |
| [MiniGrid Spatial Navigation](SimpleRL/MiniGrid_SpatialNavigation/)                               | Reinforcement Learning      | Technical Study | CNN-LSTM deep reinforcement learning for spatial navigation and exploration             | Complete |
| [Limited-Data Predictive Maintenance](SimpleTimeSeries/Limited-Data-Predictive-Maintenance/)      | Time Series                 | Experiment      | Sequence-model experiments for predictive maintenance under limited-data conditions     | Complete |
| [Kepler Exoplanet Classification](DataScience&ML/ML-For-Kepler-Exoplanet-Dataset/)                | Machine Learning            | Experiment      | Gradient boosting and automated feature engineering for exoplanet classification        | Complete |
| [Machine Translation with Seq2Seq](SimpleNLP/MachineTranslation@Seq2Seq/)                         | Natural Language Processing | Reproduction    | LSTM sequence-to-sequence implementation for machine translation                        | Complete |

> The complete project archive is organized by domain below.

---

<a id="project-collections"></a>

## Project Collections

<table>
<tr>
<td width="50%" valign="top">

### Data Science and Machine Learning

Classical machine learning, predictive modeling, feature engineering, data analysis, and optimization experiments.

[Browse Collection →](DataScience&ML/)

</td>
<td width="50%" valign="top">

### Natural Language Processing

Text classification, named-entity recognition, document classification, sequence modeling, and machine translation.

[Browse Collection →](SimpleNLP/)

</td>
</tr>

<tr>
<td width="50%" valign="top">

### Reinforcement Learning

Value-based reinforcement learning, deep reinforcement learning, navigation, and game-playing agents.

[Browse Collection →](SimpleRL/)

</td>
<td width="50%" valign="top">

### Recommender Systems

Content-based recommendation, geographic recommendation, ranking, and retrieval experiments.

[Browse Collection →](SimpleRecommender/)

</td>
</tr>

<tr>
<td width="50%" valign="top">

### Time-Series Learning

Forecasting, predictive maintenance, multivariate modeling, and sequence-model comparisons.

[Browse Collection →](SimpleTimeSeries/)

</td>
<td width="50%" valign="top">

### Graph Learning

Graph neural networks, node classification, graph representation learning, and scientific-network analysis.

[Browse Collection →](SimpleGraphsDL/)

</td>
</tr>

<tr>
<td width="50%" valign="top">

### LLM and LangChain Applications

Retrieval, context engineering, local language models, PDF processing, and document-grounded applications.

[Browse Collection →](LangchainAI/)

</td>
<td width="50%" valign="top">

### Chatbot Development

Small conversational AI systems combining neural networks, NLP pipelines, and intent-based interaction.

[Browse Collection →](DevChatBots/)

</td>
</tr>
</table>

---

## Data Science and Machine Learning

[Open Directory →](DataScience&ML/)

| Project                                                                                              | Description                                                                                                      | Type               | Year |
| ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------ | ---: |
| [Residual-Aware Rental Price Prediction](DataScience&ML/ResidualAwareML-RentalHousePricePredict/)    | Rental price prediction using a residual-aware learning module; presented at the 2025 TANet Conference in Taiwan | Publication        | 2025 |
| [Rain Predictor](DataScience&ML/RainPredictor/)                                                      | Weather prediction project developed as a machine learning midterm report                                        | Coursework         | 2024 |
| [Spam Classifier with Naive Bayes](DataScience&ML/SpamClassifier_%20NaiveBayes/)                     | Classical probabilistic text classification using Naive Bayes                                                    | Experiment         |    — |
| [Laptop Price Data Analysis](DataScience&ML/LaptopPriceDS/)                                          | Exploratory data analysis and laptop price modeling                                                              | Data Analysis      |    — |
| [Machine Learning for the Kepler Exoplanet Dataset](DataScience&ML/ML-For-Kepler-Exoplanet-Dataset/) | Gradient boosting and automated feature engineering using single-neuron transformations                          | Experiment         |    — |
| [Fox Optimizer for Neural Networks](DataScience&ML/FoxOptimizer_for_NeuralNetwork/)                  | Hyperparameter-search experiment using the Fox optimization algorithm                                            | Optimization Study |    — |

---

## Natural Language Processing

[Open Directory →](SimpleNLP/)

| Project                                                                       | Description                                                                      | Type         |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------ |
| [IMDb Sentiment Classifier](SimpleNLP/IMDB_Binary_Classifier/)                | Binary sentiment classification on IMDb reviews                                  | Experiment   |
| [AG News Classification](SimpleNLP/AGNewsClassification/)                     | News-topic classification using embeddings and an LSTM                           | Experiment   |
| [Reuters Document Classification](SimpleNLP/ReutersDocClassification/)        | Multi-class document classification using the Reuters dataset                    | Experiment   |
| [Script-to-Order Named-Entity Recognition](SimpleNLP/SimpleNER_Script2Order/) | Lightweight NER pipeline for converting text scripts into structured order lists | Prototype    |
| [PyTorch NLP Library](SimpleNLP/NLP_Lib_forDL/)                               | Reusable NLP utilities and components for deep-learning experiments              | Toolkit      |
| [Machine Translation with Seq2Seq](SimpleNLP/MachineTranslation@Seq2Seq/)     | LSTM encoder-decoder implementation for sequence-to-sequence translation         | Reproduction |

---

## Reinforcement Learning

[Open Directory →](SimpleRL/)

| Project                                                             | Description                                                                 | Type                       |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------- |
| [MiniGrid Spatial Navigation](SimpleRL/MiniGrid_SpatialNavigation/) | CNN-LSTM deep reinforcement learning for spatial navigation and exploration | Technical Study            |
| [Greedy Snake with DQN](SimpleRL/GreedySnake_MLP_DQN/)              | Deep Q-Network agent using an MLP to play a Snake-style game                | Experiment                 |
| [Maze Solving with Q-Learning](SimpleRL/Maze_QLearning/)            | Tabular Q-learning implementation for a simple maze environment             | Educational Implementation |

---

## Recommender Systems

[Open Directory →](SimpleRecommender/)

| Project                                                                                               | Description                                                  | Type       |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ---------- |
| [City Recommendation by Location](SimpleRecommender/Cities_Recommendation_By_Location/)               | Geographic recommendation based on location-related features | Prototype  |
| [Content-Based Movie Recommendation](SimpleRecommender/Content_Based_with_TFIDF_MovieRecommendation/) | Movie recommendation using TF-IDF similarity                 | Experiment |

---

## Time-Series Learning

[Open Directory →](SimpleTimeSeries/)

| Project                                                                                       | Description                                                                            | Type                   |
| --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------- |
| [Limited-Data Predictive Maintenance](SimpleTimeSeries/Limited-Data-Predictive-Maintenance/)  | Evaluation of sequence models for predictive maintenance under limited-data conditions | Technical Study        |
| [Multivariate Temperature Forecasting](SimpleTimeSeries/Time-Series-Temperature-Forecasting/) | Experiments with multivariate models for temperature forecasting                       | Experiment             |
| [Amazon Stock Price Forecasting](SimpleTimeSeries/LSTM_Amazon_Stock_Forecasting/)             | LSTM-based time-series forecasting for Amazon stock prices                             | Educational Experiment |

---

## Graph Learning

[Open Directory →](SimpleGraphsDL/)

| Project                                                                        | Description                                                                  | Type       |
| ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- | ---------- |
| [Cora Scientific Publication Classification](SimpleGraphsDL/CoRAwithGraphsNN/) | Node classification on the Cora citation network using graph neural networks | Experiment |

---

## LangChain and LLM Applications

[Open Directory →](LangchainAI/)

| Project                                                                                          | Description                                                                    | Type      |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | --------- |
| [Pizza Review Reading Bot](LangchainAI/Ollama/Ollama_Context_Engineering_UsingCSV/)              | CSV-grounded review assistant using LangChain and Ollama                       | Prototype |
| [PDF Context Engineering with Ollama](LangchainAI/Ollama/Ollama_Context_Engineering_Using_PDFs/) | Local document-question-answering experiment using PDF content about red foxes | Prototype |

---

## Chatbot Development

[Open Directory →](DevChatBots/)

| Project                                                           | Description                                                                       | Type      |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------- |
| [Small Chatbot Experiment 001](DevChatBots/ChatBot-small-ex-001/) | Lightweight chatbot combining a neural network with an NLP preprocessing pipeline | Prototype |

---

## Project Classification

| Label                          | Meaning                                                             |
| ------------------------------ | ------------------------------------------------------------------- |
| **Publication**                | Work associated with a paper, conference, or formal academic output |
| **Technical Study**            | Structured investigation of a specific method or technical question |
| **Experiment**                 | Small-scale implementation or controlled model exploration          |
| **Reproduction**               | Reimplementation of an established architecture or method           |
| **Prototype**                  | Early proof of concept for a system or application                  |
| **Coursework**                 | Project originally developed for a university course                |
| **Educational Implementation** | Fundamental method implemented primarily for learning               |
| **Toolkit**                    | Reusable utilities, modules, or supporting code                     |
| **Data Analysis**              | Exploratory data analysis and statistical investigation             |
| **Optimization Study**         | Experiment involving search or optimization algorithms              |

---

<a id="repository-guide"></a>

## Repository Guide

The repository is organized primarily by AI domain:

```text
TinyAI/
├── DataScience&ML/
├── SimpleNLP/
├── SimpleRL/
├── SimpleRecommender/
├── SimpleTimeSeries/
├── SimpleGraphsDL/
├── LangchainAI/
├── DevChatBots/
├── arctic-fox.jpg
└── README.md
```

Each project may contain a different combination of:

```text
project-name/
├── README.md
├── requirements.txt
├── notebooks/
├── data/
├── src/
├── models/
├── results/
└── assets/
```

Because these projects were developed at different stages, repository structure and documentation depth may vary.


## Getting Started

Clone the repository:

```bash
git clone https://github.com/JiaenSuen/TinyAI.git
cd TinyAI
```

Navigate to a selected project:

```bash
cd path/to/project
```

Install project-specific dependencies when available:

```bash
pip install -r requirements.txt
```

Refer to the individual project README for dataset preparation, configuration, and execution instructions.

---

## Technologies

Technologies used across TinyAI may include:

* Python
* PyTorch
* TensorFlow
* Scikit-learn
* Pandas
* NumPy
* Matplotlib
* NetworkX
* Gymnasium
* LangChain
* Ollama
* Jupyter Notebook

Dependencies vary by project.

---

## Notes and Disclaimer

* TinyAI is an experimental archive rather than a single production-ready software package.
* Some projects are educational implementations or coursework.
* Some datasets and pretrained models are provided by third parties and remain subject to their original licenses.
* Financial forecasting projects are technical demonstrations and should not be interpreted as financial advice.
* Results may vary due to hardware, random initialization, library versions, and dataset preprocessing.

---

## About the Author

**Jiaen Suen**

Computer science student interested in artificial intelligence, computer vision, machine learning, deep learning, and intelligent systems.

* GitHub: [@JiaenSuen](https://github.com/JiaenSuen)
* Email: [Add professional email]
* Website: [Add personal website]
* Curriculum Vitae: [Add CV link]
* Google Scholar: [Add Google Scholar link]

---

## License

Add the applicable repository license here.

```text
MIT License
```

Individual datasets, pretrained models, papers, and third-party implementations remain subject to their respective licenses.

---

<div align="center">

### Small Experiments, Continuous Learning

TinyAI documents practical exploration across different areas of artificial intelligence—one focused implementation at a time.

</div>
