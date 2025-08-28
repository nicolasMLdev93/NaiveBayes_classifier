from faker import Faker
import random

spam_words = ["Free", "Win", "Prize", "Click", "Buy now"]
ham_words = ["Meeting", "Report", "Project", "Schedule", "Lunch"]

fake = Faker()

def get_email(spam, ham):
    random_index = random.randint(0, 10)
    word_list = spam if random_index % 2 == 0 else ham
    # Email aleatorio
    sentence = " ".join(random.choices(word_list, k=random.randint(4, 7)))
    return sentence

fake_email = get_email(spam_words, ham_words)
print(fake_email)
