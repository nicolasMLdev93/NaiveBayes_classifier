from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NB_class:
    def __init__(self):
        self.model = MultinomialNB()
    def train_model(self, x_train, y_train):
        # Entrenamiento del modelo
        self.model.fit(x_train, y_train)
    def predict_spam(self, x_test):
        # Predicción sobre los datos de prueba
        return self.model.predict(x_test)
    def predict_prob(self, x_test):
        # Probabilidad de pertenecer a c/u de las clases => spam / no_spam
        return self.model.predict_proba(x_test)  
    def get_accuracy(self, x_test, y_test):
        # Calcula la exactitud (accuracy) del modelo
        y_pred = self.model.predict(x_test)
        return accuracy_score(y_test, y_pred)  
    def get_precision(self, x_test, y_test):
        # Calcula la precisión del modelo
        y_pred = self.model.predict(x_test)
        return precision_score(y_test, y_pred, average='binary') 
        # Métricas:
    def get_recall(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        return recall_score(y_test, y_pred, average='binary')  
    def get_f1(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        return f1_score(y_test, y_pred, average='binary')
