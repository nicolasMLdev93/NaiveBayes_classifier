import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.naivebayes_class import NB_class
from emails.testing_email import fake_email_vector

# Dataset
dataset = pd.read_csv('../data/clean_ds.csv')

def get_features(dataset):
    X = dataset["text"]   # Emails
    Y = dataset["label_num"]  # Etiquetas
    return X, Y

X, Y = get_features(dataset)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Vectorizer
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(X_train)
x_test = vectorizer.transform(X_test)

# Modelo
classifier_instance:NB_class = NB_class()
classifier_instance.train_model(x_train, y_train)

# Email falso
fake_email_vectorized = vectorizer.transform([fake_email_vector])

y_pred = classifier_instance.predict_spam(fake_email_vectorized)
y_proba = classifier_instance.predict_prob(fake_email_vectorized)

print(f"Email generado: {fake_email_vector}")
print(f"Predicción: {'es spam' if y_pred[0] == 1 else 'no es spam'}")
print(f"Probabilidad de SPAM: {y_proba[0][1]*100:.2f}%")
