import spacy
from spacy.util import minibatch
from spacy.training.example import Example
import random

train_data = [

    ("I want to order 2 cheeseburgers",
     {"entities": [(16, 17, "QUANTITY"), (18, 31, "PRODUCT")]}),

    ("Can I have 3 fried chickens",
     {"entities": [(12, 13, "QUANTITY"), (14, 28, "PRODUCT")]}),

    ("Please add 1 large pizza to my order",
     {"entities": [(11, 12, "QUANTITY"), (13, 18, "SIZE"), (19, 24, "PRODUCT")]}),

 
    ("What is the price of beef noodles",
     {"entities": [(21, 33, "PRODUCT")]}),

    ("How much does the seafood pasta cost",
     {"entities": [(18, 32, "PRODUCT")]}),

    ("The price of milk tea is 5 dollars",
     {"entities": [(13, 22, "PRODUCT"), (26, 35, "PRICE")]}),


    ("Do you serve breakfast pancakes",
     {"entities": [(13, 22, "MEAL_TIME"), (23, 31, "PRODUCT")]}),

    ("I would like a dinner steak",
     {"entities": [(15, 21, "MEAL_TIME"), (22, 27, "PRODUCT")]}),


    ("Order 2 burgers and 1 french fries",
     {"entities": [
         (6, 7, "QUANTITY"),
         (8, 15, "PRODUCT"),
         (20, 21, "QUANTITY"),
         (22, 34, "PRODUCT")
     ]}),

    ("Can I get 3 cups of coffee for 10 dollars",
     {"entities": [
         (10, 11, "QUANTITY"),
         (20, 26, "PRODUCT"),
         (31, 41, "PRICE")
     ]}),
]



nlp = spacy.load("en_core_web_md")

if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')
else:
    ner = nlp.get_pipe('ner')


for _ , annotations in train_data:
    for ent in annotations['entities']:
        if ent not in ner.labels:
            ner.add_label(ent[2])


other_pipes = [ pipe for pipe in nlp.pipe_names if pipe != 'ner']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.initialize()


    epochs = 50
    for epoch in range(epochs):
        losses = {  }
        batches = minibatch(train_data,size=2)
        for batch in batches:
            examples = []
            for text,annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                examples.append(example)
            nlp.update(examples,drop=0.5,losses=losses)
        print (f"Epoch {epoch+1} : Losses : {losses}")

nlp.to_disk('custom_ner_model')
trained_nlp = spacy.load("custom_ner_model")


test_texts = [
    "I want 2 burgers and 1 pizza",
    "How much is the seafood pasta",
    "Order 3 milk tea for 15 dollars",
    "Do you have breakfast coffee",
    "I would like a large steak",
]

for text in test_texts:
    doc = trained_nlp(text)
    print(f'Text : {text}')
    print(f"Entities : {[(ent.text,ent.label_) for ent in doc.ents]}")
    print()

 


''' Result :

Epoch 50 : Losses : {'ner': np.float32(4.412279)}



Text : I want 2 burgers and 1 pizza
Entities : [('2', 'QUANTITY'), ('burgers', 'PRODUCT'), ('1', 'QUANTITY'), ('pizza', 'PRODUCT')]

Text : How much is the seafood pasta
Entities : []

Text : Order 3 milk tea for 15 dollars
Entities : []

Text : Do you have breakfast coffee
Entities : [('breakfast', 'MEAL_TIME'), ('coffee', 'PRODUCT')]

Text : I would like a large steak
Entities : [('large', 'MEAL_TIME'), ('steak', 'PRODUCT')]

'''