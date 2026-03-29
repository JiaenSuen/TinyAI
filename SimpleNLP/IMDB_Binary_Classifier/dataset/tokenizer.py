# Tokenizer
import re
def simple_tokenizer(text):
    text = re.sub(r'<.*?>', '', text) # remove html mark
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)

    return tokens