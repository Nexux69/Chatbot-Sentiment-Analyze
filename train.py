#Import Libraries
import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nltk
nltk.download('punkt_tab')
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
#Global variables
DATA = {}
MODEL = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Hyper-parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
hidden_size = 8
X_train = []
y_train = []
train_loader = None
output_size = 0
input_size = 0
LossHistory = []
AccuracyHistroy = []
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
#1
def loadDataForTraining(fileName:str):
    global DATA
    with open(r"C:\Users\DELL\Desktop\faiz\Chatbot\Chatbot\intents.json", 'r') as f:
        intents = json.load(f)
    all_words = []
    tags = []
    xy = []
    # loop through each sentence in our intents patterns
    for intent in intents['intents']:
        tag = intent['tag']
        # add to tag list
        tags.append(tag)
        for pattern in intent['patterns']:
            # tokenize each word in the sentence
            w = tokenize(pattern)
            # add to our words list
            all_words.extend(w)
            # add to xy pair
            xy.append((w, tag))
    # stem and lower each word
    ignore_words = ['?', '.', '!']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    # remove duplicates and sort
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    print(len(xy), "patterns")
    print(len(tags), "tags:", tags)
    print(len(all_words), "unique stemmed words:", all_words)
    DATA['all_words'] = all_words
    DATA['tags'] = tags
    DATA['xy'] = xy
#2
def processDataForTraining():
    global  X_train, y_train, train_loader
    global output_size, input_size
    # create training data
    X_train = []
    y_train = []
    all_words = DATA['all_words']
    tags = DATA['tags']
    xy = DATA['xy']
    for (pattern_sentence, tag) in xy:
        # X: bag of words for each pattern_sentence
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index(tag)
        y_train.append(label)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    output_size = len(tags)
    input_size = len(X_train[0])
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    print("Data has been processed")
#3
def trainTheModel():
    print("Started Training Model")
    global MODEL,AccuracyHistroy,LossHistory
    MODEL = NeuralNet(input_size, hidden_size, output_size).to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=learning_rate)
    # Train the model
    for epoch in range(num_epochs):
        accSum = 0
        lossSum = 0
        total_samples = 0
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            # Forward pass
            outputs = MODEL(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)
             # Compute Accuracy
            _, predicted = torch.max(outputs, 1)
            accSum += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #update model training history
        LossHistory.append(loss.item())
        AccuracyHistroy.append(accSum/total_samples)
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accSum / total_samples:.4f}')
    print(f'Final loss: {loss.item():.4f}, accuracy: {accSum / total_samples:.4f}')
#4
def evaluateModel():
    # summarize history for accuracy
    Accuracyxs = [x for x in range(len(AccuracyHistroy))]
    plt.plot(Accuracyxs, AccuracyHistroy)
    plt.title("Accuracy")
    plt.show()
    # summarize history for loss
    Lossxs = [x for x in range(len(LossHistory))]
    plt.plot(Accuracyxs, LossHistory)
    plt.title("Loss")
    plt.show()
#5
def saveModel():
    Modeldata = {
    "model_state": MODEL.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": DATA['all_words'],
    "tags": DATA['tags']
    }
    FileName = r"C:\Users\DELL\Desktop\faiz\Chatbot\Chatbot\ChatBot.pth"
    torch.save(Modeldata, FileName)
    print(f'Training complete. Model saved to {FileName}')
##Helper Functions and classes
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
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
loadDataForTraining('intents.json')
processDataForTraining()
trainTheModel()
evaluateModel()
saveModel()
exit()