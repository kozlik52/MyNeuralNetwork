import numpy as np
from neural_network import NeuralNetwork
import mlflow


def evaluate_model(nn, X_train, y_train, X_test, y_test):
    """
    Оценка точности нейронной сети.

    Аргументы:
        nn: Экземпляр NeuralNetwork.
        X_train, y_train: Обучающие данные и метки.
        X_test, y_test: Тестовые данные и метки.

    Возвращает:
        float: Точность на тестовых данных.
    """
    nn.train(X_train, y_train, epochs=100, batch_size=32)
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
    return accuracy


def automl_grid_search(X_train, y_train, X_test, y_test, input_size, output_size):
    """
    Выполняет поиск по сетке для нахождения лучших гиперпараметров.

    Аргументы:
        X_train, y_train, X_test, y_test: Разделенные данные.
        input_size (int): Количество входных признаков.
        output_size (int): Количество выходных классов.

    Возвращает:
        best_params: Лучшие найденные гиперпараметры.
        best_accuracy: Лучшая точность на тестовых данных.
    """
    # Определение сетки гиперпараметров
    learning_rates = [0.001, 0.01, 0.1]
    hidden_layers = [[64], [128], [64, 32], [128, 64]]
    epochs = [100, 200]
    batch_sizes = [16, 32]

    best_accuracy = 0
    best_params = {}

    for lr in learning_rates:
        for hidden in hidden_layers:
            for epoch in epochs:
                for batch_size in batch_sizes:
                    with mlflow.start_run():
                        # Логирование параметров
                        mlflow.log_param("learning_rate", lr)
                        mlflow.log_param("hidden_layers", hidden)
                        mlflow.log_param("epochs", epoch)
                        mlflow.log_param("batch_size", batch_size)

                        # Создание и обучение модели
                        layer_sizes = [input_size] + hidden + [output_size]
                        nn = NeuralNetwork(layer_sizes, learning_rate=lr)
                        accuracy = evaluate_model(nn, X_train, y_train, X_test, y_test)

                        # Логирование метрик
                        mlflow.log_metric("accuracy", accuracy)

                        # Обновление лучших параметров
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_params = {
                                "learning_rate": lr,
                                "hidden_layers": hidden,
                                "epochs": epoch,
                                "batch_size": batch_size
                            }

    return best_params, best_accuracy