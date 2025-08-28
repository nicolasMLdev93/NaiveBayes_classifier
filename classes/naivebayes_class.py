from sklearn.naive_bayes import MultinomialNB

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
    