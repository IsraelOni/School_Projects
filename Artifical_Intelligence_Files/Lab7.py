import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split, KFold
import math
from sklearn.metrics import classification_report
"""Title: Naive Bayes Classifier for Email Spam Filtering
Author: Israel Oni
Student ID: 1675980
Date: April 2, 2024
Project Description: To build a spam filter for emails using a Naive Bayes classifier. 

This project involves implementing the algorithm from scratch, 
splitting the dataset with train_test_split, using cross-validation, and evaluating the model's performance.
Dataset: SMS Spam Collection from the UCI Machine Learning Repository."""
def preprocess_text(text_array):
    """Remove punctuation and convert text to lowercase."""
    translator = str.maketrans('', '', string.punctuation)
    processed_data = [sentence.translate(translator).lower().replace('\t', ' ').split() for sentence in text_array]
    return processed_data

def calculate_vocabulary_size(processed_data):
    """Calculate unique vocabulary size in the dataset."""
    flattened_array = [item for sublist in processed_data for item in sublist]
    unique_strings = list(set(flattened_array))
    return len(unique_strings), unique_strings

def calculate_word_frequencies(dataset):
    """Count the frequency of each word in the dataset."""
    word_count = {}
    for word in dataset:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count

def categorize_words(text_array, labels):
    """Separate words into spam and ham categories based on the labels."""
    spam_words = []
    ham_words = []
    for sublist, label in zip(text_array, labels):
        for word in sublist:
            if label == 'spam':
                spam_words.append(word)
            elif label == 'ham':
                ham_words.append(word)
    return spam_words, ham_words

def calculate_conditional_probabilities(X_train, Y_train):
    """Calculate conditional probabilities of words being in spam and ham categories."""
    processed_train = preprocess_text(X_train)
    vocab_length, vocab_list = calculate_vocabulary_size(processed_train)

    spam_words, ham_words = categorize_words(processed_train, Y_train)
    ham_word_counts = calculate_word_frequencies(ham_words)
    spam_word_counts = calculate_word_frequencies(spam_words)

    spam_cond_prob = {}
    ham_cond_prob = {}

    for word in vocab_list:
        spam_cond_prob[word] = ((spam_word_counts.get(word, 0) + 1) / (len(spam_words) + vocab_length))
        ham_cond_prob[word] = ((ham_word_counts.get(word, 0) + 1) / (len(ham_words) + vocab_length))

    return spam_cond_prob, ham_cond_prob

def calculate_prior_probabilities(labels):
    """Calculate the prior probabilities of messages being spam or ham."""
    spam_count = labels.count('spam')
    ham_count = labels.count('ham')
    total_count = len(labels)
    return ham_count / total_count, spam_count / total_count

def log_probability(probability):
    """Calculate the natural logarithm of a probability."""
    return math.log(probability) if probability != 0 else float('-inf')

def predict_spam_or_ham(messages, spam_cond_prob, ham_cond_prob, ham_prior, spam_prior):
    """Predict whether each message is spam or ham."""
    predicted_labels = []
    processed_test = preprocess_text(messages)

    for sentence in processed_test:
        spam_prob = log_probability(spam_prior) + sum(log_probability(spam_cond_prob.get(word, 1/(len(spam_cond_prob)+1))) for word in sentence)
        ham_prob = log_probability(ham_prior) + sum(log_probability(ham_cond_prob.get(word, 1/(len(ham_cond_prob)+1))) for word in sentence)

        if spam_prob > ham_prob:
            predicted_labels.append('spam')
        else:
            predicted_labels.append('ham')

    return predicted_labels

if __name__ == "__main__":
    # Load dataset
    dataset = pd.read_csv("SMSSpamCollection", sep='\t', header=None, names=['Label', 'SMS'])
    labels = dataset['Label'].tolist()
    messages = dataset['SMS'].tolist()
    # K-Fold cross-validation setup
    k_fold = KFold(n_splits=7, shuffle=True, random_state=42)

    # Perform K-Fold cross-validation
    for train_idx, test_idx in k_fold.split(messages):
        X_train, X_test = [messages[i] for i in train_idx], [messages[i] for i in test_idx]
        y_train, y_test = [labels[i] for i in train_idx], [labels[i] for i in test_idx]

        # Train model and predict
        spam_cond_prob, ham_cond_prob = calculate_conditional_probabilities(X_train, y_train)
        ham_prior, spam_prior = calculate_prior_probabilities(y_train)
        predictions = predict_spam_or_ham(X_test, spam_cond_prob, ham_cond_prob, ham_prior, spam_prior)

        # Output classification report
        print(classification_report(y_test, predictions))