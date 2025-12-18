# Named Entity Recognition
# John goes for a walk in Berlin.
# Person - > John
# Location - > Berlin

import spacy


texts = [
    "John goes for a walk in Berlin. ",
    "Mike is going to the store",
    "Elon Musk is the CEO of Tesla",
    "what is the price of 20 bananas ? ",
    "We will go to Park",
    "how much are 12 cakes?",
    "I'd like to order a cheeseburger with fries and a large Coke.",
    "Can I have the grilled salmon with vegetables and a side of mashed potatoes?",
    "We'll take two Margherita pizzas and a bottle of sparkling water.",
    "Please bring me the chicken curry with rice and naan bread.",
    "For dessert, I'd like the chocolate cake and a cup of coffee."
]


nlp = spacy.load("en_core_web_md")

#ner_label = nlp.get_pipe('ner').labels
 
 

docs = [nlp(text) for text in texts]

for doc in docs:
    entities  = []
    for ent in doc.ents:
        entities.append((ent.text,ent.label_))

    print(entities)

"""
[('John', 'PERSON'), ('Berlin', 'GPE')]
[('Mike', 'PERSON')]
[('Elon Musk', 'PERSON'), ('Tesla', 'GPE')]
[('20', 'CARDINAL')]
[('Park', 'GPE')]
[('12', 'CARDINAL')]
[('Coke', 'ORG')]
[]
[('two', 'CARDINAL')]
[]
[]

"""
# In the Result of this code, as you can see 
# Number of Item like "two","12","20" could be able to Recognized as 'CARDINAL'
# Person of Name  can also be Recognized, Place can be also do but not clearly classify...
# the problem is items like  coffee,banana,food aren't be able to be Recognized,
# so it have to be trained to fit on other tasks we want to apply.