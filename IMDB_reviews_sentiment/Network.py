import numpy as np
import math
import random
import h5py
"""
手写的BP神经网络
"""

# sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# sigmoid 导数
def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

# 误差计算
def cost_derivative(output_activations, y):
    return output_activations - y


class BPNetwork:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def freed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def evaluate(self, test_data):
        # 获得预测结果
        test_results = [(np.argmax(self.freedforward(x)), y)
                        for (x, y) in test_data]
        # 返回正确识别的个数
        return sum(int(x == y) for (x, y) in test_results)

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        else:
            n_test = 0
        n = training_data.size
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if n_test != 0:
                print("Epoch {0}: {1} / {2}".format(j, self.evauate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backdrop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backdrop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        # 乘于前一层的输出值
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            # 从倒数第 l 层开始更新，-l 是 python 中特有的语法表示从倒数第 l 层开始计算
            # 下面这里利用 l+1 层的 δ 值来计算 l 的 δ 值
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

if __name__ == '__main__':
    '''进行预测'''
    data = h5py.File('data.h5', 'r')
    x_train = data['train_reviews']
    y_train = data['train_labels']
    test = data['test_reviews']

    net = BPNetwork([300, 250, 1])
    net.SGD(x_train, 30, 10, 3.0, test_data=y_train)

    predict = net.predict(test_set)
    ans = []
    for i in predict:
        if i > 0.5:
            ans.append('positive')
        else:
            ans.append('negative')

    sub = pd.read_csv('submission.csv')

    sub['sentiment'] = ans
    sub.to_csv('subb_new.csv', index=False)
