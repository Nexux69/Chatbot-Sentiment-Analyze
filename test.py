sentence = None
results = None
def text_prompt(msg):
  try:
    return input(msg)
  except NameError:
    return input(msg)
# import libraries
import numpy as np
import random
import json
import torch
import torch.nn as nn
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('punkt_tab')
#0 Create model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # no activation and no softmax at the end
        return out
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# requires intents.json and chatbot.pth file in path
with open(r"C:\Users\DELL\Desktop\faiz\Chatbot\Chatbot\intents.json", 'r') as json_data:
    intents = json.load(json_data)
FILE = r"C:\Users\DELL\Desktop\faiz\Chatbot\Chatbot\ChatBot.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
# function for preprcessing
def loadDataForPrediction(data):
    global processed_data
    sentence = tokenize(data)
    bagOfWordsVec = bag_of_words(sentence, all_words)
    wordVec = bagOfWordsVec.reshape(1, bagOfWordsVec.shape[0])
    processed_data = torch.from_numpy(wordVec).to(device)
def getPredictionsFromModel():
    global result
    output = model(processed_data)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents[r"C:\Users\DELL\Desktop\faiz\Chatbot\Chatbot\intents.json"]:
            if tag == intent["tag"]:
                result = random.choice(intent['responses'])
    else:
        result = "I do not understand..."
def getResults():
    return result
###Helper functions
stemmer = PorterStemmer()
def tokenize(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)
def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag
while True:
  sentence = text_prompt('you:')
  if sentence == 'quit':
    break
  loadDataForPrediction(sentence)
  getPredictionsFromModel()
  results = getResults()
  print(''.join([str(x) for x in ['Rent-A-car:{', result, '}']]))