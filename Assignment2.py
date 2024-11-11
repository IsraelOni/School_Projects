import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


def preprocess_text(text):
  # Remove special characters and digits
  text = re.sub(r'[^A-Za-z\s]', '', text)

  # Convert to lowercase
  text = text.lower()

  # Tokenize into sentences
  sentences = re.split(r'\s*\.\s*|\s*!\s*|\s*\?\s*|\s*:\s*', text)

  # Tokenize each sentence into words
  tokenized_sentences = []
  for sentence in sentences:
      if sentence:  # Ensures the sentence is not empty
          tokenized_sentences.append(sentence.split())

  return tokenized_sentences


def build_vocabulary(sentences):
  vocabulary = {}
  for sentence in sentences:
      for word in sentence:
          if word in vocabulary:
              vocabulary[word] += 1
          else:
              vocabulary[word] = 1
  return vocabulary



def generate_context_target_pairs(tokenized_sentences, window_size):
    contexts = []
    targets = []
    for sentence in tokenized_sentences:
        for idx, word in enumerate(sentence):
            start_index = max(0, idx - window_size)
            end_index = min(len(sentence), idx + window_size + 1)

            # Initialize context for each target word
            context = []
            for i in range(start_index, end_index):
                if i != idx:
                    context.append(sentence[i])

            target = word
            if len(context) == window_size*2:
                contexts.append(context)
                targets.append(target)
    return contexts, targets

def one_hot_encode(word, vocab):
    # Create a zero vector with length equal to the size of the vocabulary
    vector = np.zeros(len(vocab))
    # Set the index corresponding to the word to 1
    counter = 0
    for index, key in enumerate(vocab):
        if key == word:
            break
        counter += 1

    vector[counter] = 1
    return vector

#def forward(context_idxs, theta):
#    m = embeddings[context_idxs].reshape(1, -1)
#    n = linear(m, theta)
#    o = log_softmax(n)
#    return m, n, o



def linear(m, theta):
    w = theta
    return m.dot(w)

def log_softmax(x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

def NLLLoss(logs, targets):
    out = logs[range(len(targets)), targets]
    return -out.sum()/len(out)


def log_softmax_crossentropy_with_logits(logits, target):
    out = np.zeros_like(logits)
    out[np.arange(len(logits)), target] = 1

    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)

    return (- out + softmax) / logits.shape[0]


def backward(preds, theta, target_idxs):
    m, n, o = preds

    dlog = log_softmax_crossentropy_with_logits(n, target_idxs)
    dw = m.T.dot(dlog)
    return dw
def optimize(theta, grad, lr=0.03):
    theta -= grad * lr
    return theta


#def predict(words):
#    context_idxs = np.array([word_to_ix[w] for w in words])
#    preds = forward(context_idxs, theta)
#    word = ix_to_word[np.argmax(preds[-1])]

#    return word


class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def training(contexts, targets,model,optimizer,loss_function):
    loss_history = []
    for epoch in range(200):
        total_loss = 0
        for context, target in zip(contexts, targets):
            context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
            if (len(context_idxs) == window_size*2):
                model.zero_grad()
                log_probs = model(context_idxs)
                loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / len(targets)
            loss_history.append(average_loss)

    return loss_history


def test(contexts, targets):
    predictions = []
    actuals = []
    for context, target in zip(contexts, targets):
        context_idxs = torch.tensor([word_to_ix[w] for w in context if w in word_to_ix], dtype=torch.long)
        if (len(context_idxs) == window_size*2):
            model.zero_grad()
            log_probs = model(context_idxs)
            pred = np.argmax(log_probs.detach().numpy())
            #print('prediction: {}'.format(ix_to_word[pred]))
            #print('actual: {}'.format(target))
            predictions.append(ix_to_word[pred])
            actuals.append(target)
    return predictions,actuals


def plot_loss_history(loss_history):
    """
    Visualizes the model training loss across epochs using a line plot.
    """
    plt.figure()
    plt.plot(loss_history, marker='o', linestyle='-', color='blue')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
if __name__ == "__main__":
    file_path = 'asm2_test.txt'

    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()


    tokenized_sentences = preprocess_text(data)
    vocabulary = build_vocabulary(tokenized_sentences)

    window_size = 2

    # Generate context-target pairs
    contexts, targets = generate_context_target_pairs(tokenized_sentences, window_size)

    # contexts, targets = generate_context_target_pairs(tokenized_sentences, window_size)

    # Splitting data into train and test sets directly
    context_train, context_test, target_train, target_test = train_test_split(contexts, targets, test_size=0.2,
                                                                              random_state=42)

    EMBEDDING_DIM = 10

    word_to_ix = {word: i for i, word in enumerate(vocabulary)}
    ix_to_word = {i: word for i, word in enumerate(vocabulary)}
    model = NGramLanguageModeler(len(vocabulary), EMBEDDING_DIM, window_size)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_function = nn.NLLLoss()
    plot_history = training(context_train,target_train,model,optimizer,loss_function)
    predictions,actuals = test(context_test,target_test)
    plot_loss_history(plot_history)
    print(accuracy_score(actuals, predictions))