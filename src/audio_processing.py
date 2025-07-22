import os
import numpy as np
import librosa
from scipy.io import wavfile


def load_audio_files(data_dir, sample_rate=8000, n_mfcc=13):
    """
    Загружает и предобрабатывает аудиофайлы из датасета FSDD, извлекая MFCC-признаки.

    Аргументы:
        data_dir (str): Путь к датасету.
        sample_rate (int): Целевая частота дискретизации аудио.
        n_mfcc (int): Количество MFCC-признаков для извлечения.

    Возвращает:
        features (list): Список массивов MFCC-признаков.
        labels (list): Список соответствующих меток цифр (0-9).
    """
    features = []
    labels = []

    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            # Извлечение метки цифры из имени файла (например, '0_jane_0.wav' -> 0)
            label = int(filename.split('_')[0])
            file_path = os.path.join(data_dir, filename)

            # Загрузка аудиофайла
            audio, sr = librosa.load(file_path, sr=sample_rate)

            # Извлечение MFCC-признаков
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

            # Усреднение MFCC по времени для получения вектора фиксированного размера
            mfcc_mean = np.mean(mfcc, axis=1)

            features.append(mfcc_mean)
            labels.append(label)

    return np.array(features), np.array(labels)


def preprocess_data(data_dir, test_split=0.2):
    """
    Загружает и разделяет данные на обучающую и тестовую выборки.

    Аргументы:
        data_dir (str): Путь к датасету.
        test_split (float): Доля данных для тестирования.

    Возвращает:
        X_train, X_test, y_train, y_test: Разделенные и предобработанные данные.
    """
    X, y = load_audio_files(data_dir)

    # Перемешивание данных
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Разделение на обучающую и тестовую выборки
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Нормализация признаков
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-6  # Избежание деления на ноль
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test