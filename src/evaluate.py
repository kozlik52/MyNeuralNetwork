import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import mlflow
import os


def evaluate_model(nn, X, y, run_name, save_dir="experiments"):
    """
    Оценка нейронной сети и построение матрицы соответствий.

    Аргументы:
        nn: Обученный экземпляр NeuralNetwork.
        X: Данные для оценки.
        y: Метки в формате one-hot или целочисленные.
        run_name (str): Имя запуска для логирования в MLflow.
        save_dir (str): Папка для сохранения матрицы соответствий.

    Возвращает:
        float: Точность на данных.
    """
    # Предсказания
    predictions = nn.predict(X)
    if y.ndim > 1:  # Если метки в формате one-hot
        y_true = np.argmax(y, axis=1)
    else:
        y_true = y

    # Вычисление точности
    accuracy = np.mean(predictions == y_true)

    # Построение матрицы соответствий
    cm = confusion_matrix(y_true, predictions, labels=np.arange(10))

    # Визуализация матрицы соответствий
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.arange(10), yticklabels=np.arange(10))
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Истинные метки')
    plt.title(f'Матрица соответствий ({run_name})')

    # Сохранение изображения
    os.makedirs(save_dir, exist_ok=True)
    cm_path = os.path.join(save_dir, f"confusion_matrix_{run_name}.png")
    plt.savefig(cm_path)
    plt.close()

    # Логирование в MLflow
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_artifact(cm_path)

    return accuracy