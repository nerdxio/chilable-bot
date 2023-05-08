import pandas as pd
import numpy as np
from sklearn.svm import SVC
import nltk
from nltk.stem import WordNetLemmatizer
import string

# Load the preprocessed CSV file
df = pd.read_csv('dialogs.csv', header=None, names=['context', 'response', 'intent'])
features = df.iloc[:, 3:].values
labels = df['intent'].values

# Train an SVM classifier
clf = SVC(kernel='linear', C=1.0, random_state=42)
clf.fit(features, labels)

# Define a function to preprocess a new message
def preprocess_message(message):
    lemmatizer = WordNetLemmatizer()
    message = message.lower()
    message = message.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(message)
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

# Define a function to predict the intent of a new message
def predict_intent(message):
    tokens = preprocess_message(message)
    features = np.zeros((1, len(word_features)))
    for token in tokens:
        if token in word_features:
            features[0, word_features[token]] = 1
    intent = clf.predict(features)[0]
    return intent

# Test the model on some sample messages
messages = [
    "What's the weather like today?",
    "What time does the train leave?",
    "Can you recommend a good restaurant?",
    "How do I reset my password?",
    "Thank you for your help!"
]

for message in messages:
    intent = predict_intent(message)
    print(f"Message: {message}\nIntent: {intent}\n")
