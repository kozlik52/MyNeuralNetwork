import sys
sys.path.append("C:/Users/Иван/OneDrive/Рабочий стол/TIOI/src")
import numpy as np
from src.audio_processing import preprocess_data
from src.automl import automl_grid_search
from src.train import train_final_model
from src.evaluate import evaluate_model
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

    # Загрузка и предобработка данных
    data_dir = "data/free-spoken-digit-dataset/recordings"
    X_train, X_test, y_train, y_test = preprocess_data(data_dir)

    # Преобразование меток в формат one-hot
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    # Запуск AutoML для поиска лучших гиперпараметров
    best_params, best_accuracy = automl_grid_search(X_train, y_train, X_test, y_test,
                                                    input_size=X_train.shape[1], output_size=10)
    print(f"Лучшие параметры: {best_params}")
    print(f"Лучшая точность: {best_accuracy}")

    # Обучение финальной модели
    nn = train_final_model(X_train, y_train, X_test, y_test, best_params)

    # Оценка финальной модели
    final_accuracy = evaluate_model(nn, X_test, y_test)
    print(f"Точность финальной модели: {final_accuracy}")


if __name__ == "__main__":
    main()