import numpy as np
from neural_network import NeuralNetwork
import mlflow


def train_final_model(X_train, y_train, X_test, y_test, best_params):
    """
    Обучение финальной модели с лучшими гиперпараметрами.

    Аргументы:
        X_train, y_train, X_test, y_test: Разделенные данные.
        best_params (dict): Лучшие гиперпараметры от AutoML.

    Возвращает:
        nn: Обученная нейронная сеть.
    """
    with mlflow.start_run(run_name="final_model"):
        # Логирование лучших параметров
        for key, value in best_params.items():
            mlflow.log_param(key, value)

        # Создание и обучение модели
        layer_sizes = [X_train.shape[1]] + best_params["hidden_layers"] + [y_train.shape[1]]
        nn = NeuralNetwork(layer_sizes, learning_rate=best_params["learning_rate"])
        nn.train(X_train, y_train, epochs=best_params["epochs"], batch_size=best_params["batch_size"])

        # Оценка и логирование точности
        predictions = nn.predict(X_test)
        accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
        mlflow.log_metric("final_accuracy", accuracy)

        return nn