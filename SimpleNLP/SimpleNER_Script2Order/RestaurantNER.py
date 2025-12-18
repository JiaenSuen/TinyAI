import spacy
from spacy.util import minibatch
from spacy.training.example import Example
from spacy.scorer import Scorer
import re
import random
from MacDATA import MENU, test_data_raw
 
products = list(MENU.keys())

templates = [
    "I'd like {quantity} {product}.",
    "Can I get {quantity} {product} and {quantity2} {product2}?",
    "Hello, I want a {product}, please.",
    "Give me {quantity} {product}.",
    "One {product} and {quantity} {product2}, thanks.",
    "I'll have {quantity} {product}.",
    "Order: {quantity} {product} and a {product2}.",
    "Hi, {quantity} {product} for me.",
    "I need {quantity} {product} and {quantity2} {product2} right away.",
    "Just {quantity} {product}, nothing else.",
    "Hey, can you add {quantity} {product} to my order?",
    "I’d love two {product} and one {product2}.",
    "Please prepare {quantity} {product}, and {quantity2} {product2}.",
    "Quick order: a {product}, plus {quantity} {product2}.",
    "For lunch, {quantity} {product}, and a {product2}."
]



def generate_examples(num_examples, seed=42):
    random.seed(seed)
    data = []

    for _ in range(num_examples):
        template = random.choice(templates)

        entities = []
        cursor = 0
        text = ""

        def append_text(chunk):
            nonlocal text, cursor
            text += chunk
            cursor += len(chunk)

        def append_quantity(qty):
            nonlocal text, cursor, entities
            qty_str = str(qty)
            start = cursor
            append_text(qty_str)
            end = cursor
            entities.append((start, end, "QUANTITY"))

        def append_product(prod):
            nonlocal text, cursor, entities
            start = cursor
            append_text(prod)
            end = cursor
            entities.append((start, end, "PRODUCT"))

        if "{product2}" in template:
            product = random.choice(products)
            product2 = random.choice(products)
            quantity = random.randint(1, 5)
            quantity2 = random.randint(1, 5)

            parts = re.split(r"(\{quantity\}|\{product\}|\{quantity2\}|\{product2\})", template)

            for part in parts:
                if part == "{quantity}":
                    append_quantity(quantity)
                elif part == "{product}":
                    append_product(product)
                elif part == "{quantity2}":
                    append_quantity(quantity2)
                elif part == "{product2}":
                    append_product(product2)
                else:
                    append_text(part)

        else:
            product = random.choice(products)
            quantity = random.randint(1, 5)

            parts = re.split(r"(\{quantity\}|\{product\})", template)

            for part in parts:
                if part == "{quantity}":
                    append_quantity(quantity)
                elif part == "{product}":
                    append_product(product)
                else:
                    append_text(part)

        data.append((text, {"entities": entities}))

    return data





def make_example(nlp, text, spans):
    doc = nlp.make_doc(text)
    ents = []

    for span_text, label in spans:
        start = text.find(span_text)
        if start == -1:
            continue

        end = start + len(span_text)
        span = doc.char_span(start, end, label=label, alignment_mode="contract")

        if span is not None:
            ents.append(span)

    doc.ents = ents
    return Example.from_dict(
        doc,
        {"entities": [(e.start_char, e.end_char, e.label_) for e in ents]}
    )








train_data = generate_examples(200, seed=42)


nlp = spacy.blank("en")
ner = nlp.add_pipe("ner")

ner.add_label("QUANTITY")
ner.add_label("PRODUCT")

optimizer = nlp.initialize()

epochs = 20
for epoch in range(epochs):
    losses = {}
    random.shuffle(train_data)
    batches = minibatch(train_data, size=8)
    for batch in batches:
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        nlp.update(examples, drop=0.35, losses=losses, sgd=optimizer)
    print(f"Epoch {epoch+1}: Losses: {losses}")

nlp.to_disk('custom_ner_model')
trained_nlp = spacy.load("custom_ner_model")

scorer = Scorer()
test_examples = []

for text, spans in test_data_raw:
    example = make_example(trained_nlp, text, spans)
    test_examples.append(example)

scores = scorer.score(test_examples)

 

def parse_order(text, model, menu):
    doc = model(text)
    order_items = []
    current_quantity = 1
    for ent in doc.ents:
        if ent.label_ == "QUANTITY":
            try:
                current_quantity = int(ent.text)
            except ValueError:
                current_quantity = 1
        elif ent.label_ == "PRODUCT":
            product_name = ent.text
            if product_name in menu:
                order_items.append({
                    "product": product_name,
                    "quantity": current_quantity,
                    "price": menu[product_name]
                })
                current_quantity = 1
    total = sum(item["quantity"] * item["price"] for item in order_items)
    formatted = "\n".join([f"{item['quantity']}x {item['product']}" for item in order_items]) + f"\nTotal: ${round(total, 2)}"
    return formatted

with open('output.txt', 'w', encoding='utf-8') as f:
    f.write("NER Evaluation Scores:\n")
    f.write(f"Precision: {scores['ents_p']:.4f}\n")
    f.write(f"Recall: {scores['ents_r']:.4f}\n")
    f.write(f"F-Score: {scores['ents_f']:.4f}\n\n")
    
    for text, _ in test_data_raw:
        doc = trained_nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        order = parse_order(text, trained_nlp, MENU)

        f.write(f"Text: {text}\n")
        f.write(f"Entities: {entities}\n")
        f.write(f"Formatted Order:\n{order}\n\n")