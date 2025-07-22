import numpy as np


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Инициализация нейронной сети.

        Аргументы:
            layer_sizes (list): Список целых чисел, представляющих количество нейронов в каждом слое.
            learning_rate (float): Скорость обучения для градиентного спуска.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []

        # Инициализация весов и смещений
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def sigmoid(self, x):
        """Сигмоидная функция активации."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        """Производная сигмоидной функции."""
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def softmax(self, x):
        """Функция softmax для выходного слоя."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """
        Прямое распространение.

        Аргументы:
            X (ndarray): Входные данные.

        Возвращает:
            activations: Список активаций для каждого слоя.
        """
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            if i < len(self.weights) - 1:
                a = self.sigmoid(z)
            else:
                a = self.softmax(z)
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y, output):
        """
        Обратное распространение для вычисления градиентов.

        Аргументы:
            X (ndarray): Входные данные.
            y (ndarray): Истинные метки (в формате one-hot).
            output (ndarray): Предсказанный выход.
        """
        m = X.shape[0]
        self.d_weights = [np.zeros_like(w) for w in self.weights]
        self.d_biases = [np.zeros_like(b) for b in self.biases]

        # Ошибка выходного слоя
        error = output - y
        self.d_weights[-1] = np.dot(self.activations[-2].T, error) / m
        self.d_biases[-1] = np.sum(error, axis=0, keepdims=True) / m

        # Скрытые слои
        for i in range(len(self.weights) - 2, -1, -1):
            error = np.dot(error, self.weights[i + 1].T) * self.sigmoid_derivative(self.z_values[i])
            self.d_weights[i] = np.dot(self.activations[i].T, error) / m
            self.d_biases[i] = np.sum(error, axis=0, keepdims=True) / m

    def update_parameters(self):
        """Обновление весов и смещений с использованием градиентов."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def train(self, X, y, epochs=100, batch_size=32):
        """
        Обучение нейронной сети.

        Аргументы:
            X (ndarray): Обучающие данные.
            y (ndarray): Обучающие метки (в формате one-hot).
            epochs (int): Количество эпох обучения.
            batch_size (int): Размер мини-пакетов.
        """
        m = X.shape[0]
        for epoch in range(epochs):
            # Перемешивание данных
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Обучение по мини-пакетам
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Прямой и обратный проход
                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
                self.update_parameters()

    def predict(self, X):
        """Предсказание меток классов для входных данных."""
        output = self.forward(X)
        return np.argmax(output, axis=1)