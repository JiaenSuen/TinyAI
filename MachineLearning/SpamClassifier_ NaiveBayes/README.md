# Spam Classifier using Naive Bayes

### Data Frame
text | spam
-----|------|
0  Subject: naturally irresistible your corporate...|     1
1  Subject: the stock trading gunslinger  fanny i...|     1
2  Subject: unbelievable new homes made easy  im ...|     1
3  Subject: 4 color printing special  request add...|     1
4  Subject: do not have money , get software cds ...|     1


## Compare NLP Preprocessing Methods (Naive Bayes)

### Bag of Words
-- Vocabulary size: 7945  
**Train Accuracy : 0.8970**   
**Test Accuracy  : 0.8508**

---


### Stemming + Stop Word + TF-IDF
-- Vocabulary size: 6060  
**Train Accuracy : 0.9956**   
**Test Accuracy  : 0.9372**

---

### Bag of Words + Stop Word + **Bigram**
-- Vocabulary size: 85535  
**Train Accuracy : 0.9967**   
**Test Accuracy  : 0.9546**