import numpy as np


def evaluate_model(nn, X_test, y_test):
    """
    Оценка нейронной сети на тестовых данных.

    Аргументы:
        nn: Обученный экземпляр NeuralNetwork.
        X_test, y_test: Тестовые данные и метки.

    Возвращает:
        float: Точность на тестовых данных.
    """
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
    return accuracy