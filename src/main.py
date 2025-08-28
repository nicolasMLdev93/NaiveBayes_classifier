from sklearn.naive_bayes import MultinomialNB

class Naive_bayes:
    def __init__(self):
        self.model = MultinomialNB()
    def train_model(self,x_train,y_train):
        # Entrenamos el modelo
        self.model.fit(x_train,y_train)
    def get_class(self,x_test):
        # Obtenemos la clase de resultado de la predicción
        y_test = self.model.predict(x_test)
        return y_test