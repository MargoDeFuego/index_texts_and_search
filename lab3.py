import random
import numpy as np
from tensorflow.keras.datasets import mnist

# Нейрон
class Neuron:
    def __init__(self):
        self.weight = random.uniform(-1, 1)

    def calculate(self, x):
        return x * self.weight

    def update(self, delta):
        self.weight += delta

# Сеть из нейронов
class NeuralNetwork:
    def __init__(self, input_size):
        self.neurons = [Neuron() for _ in range(input_size)]

    def calculate(self, inputs):
        return sum(self.neurons[i].calculate(inputs[i]) for i in range(len(inputs)))

    def train(self, inputs, target, studyCoeff):
        output = self.calculate(inputs)
        error = target - output

        for i in range(len(self.neurons)):
            self.neurons[i].update(studyCoeff * error * inputs[i])

        return error


# Загрузка MNIST
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# нормализация
train_x = train_x.reshape(-1, 784) / 255.0
test_x = test_x.reshape(-1, 784) / 255.0

# Параметры
digit = 7              # цифра, которую сеть должна распознавать
studyCoeff = 0.01      # коэффициент обучения
correctCoeff = 5       # порог распознавания

net = NeuralNetwork(784)

# обучение
def train_network(epochs=1):
    print("Начало обучения...")
    for epoch in range(epochs):
        indices = np.random.permutation(len(train_x))
        
        for i in indices:
            x = train_x[i]
            y = train_y[i]
            target = 1 if y == digit else 0
            net.train(x, target, studyCoeff)
            
        print(f"Эпоха {epoch+1} завершена.")

def evaluate_accuracy():
    correct = 0
    total = len(test_x)
    
    for i in range(total):
        x = test_x[i]
        y = test_y[i]
        target = 1 if y == digit else 0
        
        output = net.calculate(x)
        prediction = 1 if output >= correctCoeff else 0
        
        if prediction == target:
            correct += 1
            
    accuracy = correct / total
    print(f"\nРеальная точность на тестовых данных: {accuracy:.3f} ({correct}/{total})")
    return accuracy

# Тестирование
def test_network():
    idx = random.randint(0, len(test_x)-1)
    x = test_x[idx]
    y = test_y[idx]

    output = net.calculate(x)
    prediction = 1 if output >= correctCoeff else 0

    print("Цифра на изображении:", y)
    print("Сеть считает:", "эта цифра — 7" if prediction == 1 else "это НЕ 7")
    print("Сырое значение сети:", output)


# Запуск
if __name__ == "__main__":
    train_network(epochs=3)     
    evaluate_accuracy()         
    test_network()              
