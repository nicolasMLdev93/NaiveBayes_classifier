from faker import Faker
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

spam_words:list[str] = ["Free", "Win", "Prize", "Click", "Buy now"]
ham_words:list[str] = ["Meeting", "Report", "Project", "Schedule", "Lunch"]

fake = Faker()

vectorizer = CountVectorizer()

def get_email(spam, ham):
    random_index = random.randint(0, 10)
    word_list = spam if random_index % 2 == 0 else ham
    # Email aleatorio
    sentence = " ".join(random.choices(word_list, k=random.randint(4, 7)))
    return sentence

fake_email_vector = get_email(spam_words, ham_words)



