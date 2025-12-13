import os
import json
import random

import nltk
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import nltk
from nltk.tokenize import word_tokenize

# NLTK
nltk.data.path.append('./nltk_data')
nltk.download('punkt'     , download_dir='./nltk_data')
nltk.download('punkt_tab' , download_dir='./nltk_data')
nltk.download('wordnet'   , download_dir='./nltk_data')
#print(word_tokenize("This is a test."))



class ChatBotModel(nn.Module):

    def __init__(self, input_size , output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    


class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings = None):

        self.model = None 
        self.intents_path = intents_path

        self.documents  = []
        self.vocabulary = []
        self.intents    = []
        self.intents_responses = {}

        self.function_mappings = function_mappings

        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]
        return words
    
 
    def bag_of_words(self,words ):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()
        if os.path.exists(self.intents_path):
            with open(self.intents_path,'r') as f:
                intents_data = json.load(f)

   

        for intent in intents_data['intents']:
            if intent['tag'] not in self.intents:
                self.intents.append(intent['tag'])
                self.intents_responses[intent['tag']] = intent['responses']

            for pattern in intent['patterns']:
                pattern_words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(pattern_words)
                self.documents.append( (pattern_words,intent['tag']) )

            self.vocabulary = sorted(set(self.vocabulary))


    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.X = np.array(bags)
        self.y = np.array(indices)


    def train_model(self,batch_size,lr,epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor,y_tensor)
        loader  = DataLoader(dataset,batch_size=batch_size,shuffle=True)

        self.model = ChatBotModel(self.X.shape[1],len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(),lr=lr)

        self.model.to('cuda')

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X , batch_y in loader:
                batch_X = batch_X.to('cuda')
                batch_y = batch_y.to('cuda')
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss  = criterion(outputs,batch_y)
                loss.backward()
                optimizer.step()
                running_loss+=loss

            print(f"Epoch {epoch+1} : Loss {running_loss/len(loader):.4f}")


    def save_model(self,model_path,dimension_path):
        torch.save(self.model.state_dict(),model_path)
        with open(dimension_path,'w') as f:
            json.dump({'input_size' : self.X.shape[1] , 'output_size' : len(self.intents)},f)

    def load_model(self,model_path,dimension_path):
        with open(dimension_path,'r') as f:
            dimension = json.load(f)
        self.model = ChatBotModel(dimension['input_size'],dimension['output_size'])
        self.model.load_state_dict(torch.load(model_path,weights_only=True))

    def process_message(self,input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag   = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag],dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions)
        predicted_intent     = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent]()

        if self.intents_responses[predicted_intent]:
            return random.choice(self.intents_responses[predicted_intent])
        else:
            return None

 

if __name__ == '__main__':
    input_mode = input("Choose mode 1. Train , 2. Chat : ")
    if input_mode == '1':
        assistant = ChatbotAssistant('intents.json')
        assistant.parse_intents()
        assistant.prepare_data()
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)
        assistant.save_model('model/chatbot_model_ann.pth', 'model/dimensions_ann.json')
        exit()

    else:

        assistant = ChatbotAssistant('intents.json' )
        assistant.parse_intents()
        assistant.load_model('model/chatbot_model_ann.pth', 'model/dimensions_ann.json')

        while True:
            message = input('Enter your message:')

            if message == '/quit':
                break

            print(assistant.process_message(message))
  