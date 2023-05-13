import nltk
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

intents = json.loads(open('chatbot.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize words and append them to words list, append (words, tag) to documents list
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize the words
lemmatizer = nltk.WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

rows = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        rows.append({'tag': intent['tag'], 'pattern': pattern, 'response': intent['responses'][0]})
df = pd.DataFrame(rows)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['tag'])

tfidf_vectorizer = TfidfVectorizer(max_features=2000, use_idf=True)
X = tfidf_vectorizer.fit_transform(df['pattern']).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)


accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
print("Model accuracy: {:.2f}%".format(accuracy * 100))


while True:
    input_message = input("You: ")
    if input_message.lower() == "quit":
        break
    input_message = [lemmatizer.lemmatize(word.lower()) for word in input_message.split()]
    input_message_vec = tfidf_vectorizer.transform([" ".join(input_message)]).toarray()
    predicted_tag = label_encoder.inverse_transform(classifier.predict(input_message_vec))[0]
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            responses = intent['responses']
            print(np.random.choice(responses))
            break
