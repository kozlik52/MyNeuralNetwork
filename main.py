import numpy as np
from src.audio_processing import preprocess_data
from src.automl import automl_grid_search
from src.train import train_final_model
from src.evaluate import evaluate_model
from src.neural_network import NeuralNetwork
import mlflow


def one_hot_encode(y, num_classes=10):
    """
    Преобразование меток в формат one-hot.

    Аргументы:
        y (ndarray): Целочисленные метки.
        num_classes (int): Количество классов.

    Возвращает:
        ndarray: Метки в формате one-hot.
    """
    return np.eye(num_classes)[y]


def main():
    # Установка эксперимента MLflow
    mlflow.set_experiment("spoken_digit_recognition")

    # Загрузка и предобработка данных для обучения и AutoML
    data_dir = "data/free-spoken-digit-dataset/recordings"
    X_train, X_test, y_train, y_test = preprocess_data(data_dir, test_split=0.2)

    # Преобразование меток в формат one-hot
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    # Загрузка всего датасета для финальной оценки
    X_full, y_full = preprocess_data(data_dir, test_split=0)
    y_full_one_hot = one_hot_encode(y_full)

    # Начальные параметры для нейронной сети
    initial_params = {
        "learning_rate": 0.01,
        "hidden_layers": [64],
        "epochs": 100,
        "batch_size": 32
    }

    # Обучение и оценка модели с начальными параметрами
    initial_layer_sizes = [X_train.shape[1]] + initial_params["hidden_layers"] + [10]
    initial_nn = NeuralNetwork(initial_layer_sizes, learning_rate=initial_params["learning_rate"])
    initial_nn.train(X_train, y_train, epochs=initial_params["epochs"], batch_size=initial_params["batch_size"])
    initial_accuracy = evaluate_model(initial_nn, X_full, y_full_one_hot, run_name="initial_params")
    print(f"Точность модели с начальными параметрами: {initial_accuracy}")

    # Запуск AutoML для поиска лучших гиперпараметров
    best_params, best_accuracy = automl_grid_search(X_train, y_train, X_test, y_test,
                                                    input_size=X_train.shape[1], output_size=10)
    print(f"Лучшие параметры: {best_params}")
    print(f"Лучшая точность AutoML: {best_accuracy}")

    # Обучение финальной модели с лучшими параметрами
    final_nn = train_final_model(X_train, y_train, X_test, y_test, best_params)

    # Оценка финальной модели на всем датасете
    final_accuracy = evaluate_model(final_nn, X_full, y_full_one_hot, run_name="final_params")
    print(f"Точность финальной модели: {final_accuracy}")


if __name__ == "__main__":
    main()