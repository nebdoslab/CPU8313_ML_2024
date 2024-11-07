import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


def forward_pass(i1, i2, W1, W2, W3, W4, W5, W6, W7, W8, b11, b12, b21, b22):
    # Hidden layer activations
    h1_input = W1 * i1 + W2 * i2 + b11
    h1 = sigmoid(h1_input)
    h2_input = W3 * i1 + W4 * i2 + b12
    h2 = sigmoid(h2_input)

    # Output layer activations
    o1_input = W5 * h1 + W6 * h2 + b21
    o1 = sigmoid(o1_input)
    o2_input = W7 * h1 + W8 * h2 + b22
    o2 = sigmoid(o2_input)

    return h1, h2, o1, o2


if __name__ == '__main__':
    # Inputs and targets
    i1 = 0.9
    i2 = 0.3
    t1 = 0.01
    t2 = 0.99

    # Weights and biases
    W1 = 0.8084867976808748
    W2 = -0.3971710674397084
    W3 = -0.1943474861616222
    W4 = 0.30188417127945927
    W5 = -0.7300018683261689
    W6 = -0.3280540326494061
    W7 = 0.514835405428898
    W8 = 0.6138722343470343
    b11 = 0.109429775200972
    b12 = 0.6062805709315309
    b21 = 0.15509964477677057
    b22 = 0.42220244974067606

    learning_rate = 0.5

    h1, h2, o1, o2 = forward_pass(i1, i2, W1, W2, W3, W4, W5, W6, W7, W8, b11, b12, b21, b22)
    print(f'h1: {h1}\nh2: {h2}\no1: {o1}\no2: {o2}')

    e1 = math.pow((t1 - o1), 2)
    e2 = math.pow((t2 - o2), 2)
    print(f'e1: {e1}\ne2: {e2}')
    et = (e1 + e2) / 2
    print(f'et: {et}')

    total_error = (((t1 - o1) ** 2) + ((t2 - o2) ** 2))/2
    print(f'Total error: {total_error}')

