import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classes.naivebayes_class import NB_class
from emails.testing_email import fake_email_vector

# Dataset
dataset = pd.read_csv('../data/clean_ds.csv')

# Separar features y target
X = dataset["text"]
Y = dataset["label_num"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Vectorizar
vectorizer = CountVectorizer()
x_train = vectorizer.fit_transform(X_train)
x_test = vectorizer.transform(X_test)

# Entrenar modelo
classifier_instance = NB_class()
classifier_instance.train_model(x_train, y_train)

# Predicciones sobre X_test
y_proba_test = classifier_instance.predict_prob(x_test)
y_pred_test = classifier_instance.predict_spam(x_test)

# Curva Precision-Recall
precision, recall, _ = precision_recall_curve(y_test, y_proba_test[:,1])

plt.figure(figsize=(8,6))
plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Curva Precision-Recall")
plt.legend()
plt.grid(True)
plt.savefig("precision_recall_curve.png")  
plt.show()

# Predicción de un email falso
fake_email_vectorized = vectorizer.transform([fake_email_vector])
y_pred_fake = classifier_instance.predict_spam(fake_email_vectorized)
y_proba_fake = classifier_instance.predict_prob(fake_email_vectorized)

print(f"Email generado: {fake_email_vector}")
print(f"Predicción del email falso: {y_pred_fake[0]}")
print(f"Probabilidad de SPAM: {y_proba_fake[0][1]*100:.2f}%")
