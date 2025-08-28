from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score

class NB_class:
    def __init__(self):
        self.model = MultinomialNB()
    def tran_model(self,x_train,y_train):
        # Entrenamiento del modelo
        self.model.fit(x_train,y_train)
        return y_train
    def predict_spam(self,x_test):
        y_test = self.model.predict(x_test)
        return y_test
    def predict_prob(self,x_test):
        # Probabilidad de pertenecer a una clase u a otra
        return self.model.predict_proba(x_test)
    def get_accu(self,x_test,y_test):
        # Predicciones correctas
        y_pred = self.model.predict(x_test)
        return accuracy_score(y_test,y_pred)
    def get_preci_score(self,x_test,y_test):
        # Valor de precisíon de la predicción
        y_pred = self.model.predict(x_test)
        return precision_score(y_test,y_pred)