import numpy as np
import matplotlib.pylab as plt


#계단  함수
def step_function(x):
    return np.array(x > 0, dtype=np.int)


#시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


#ReLU 함수
def relu(x):
    return np.maximum(0, x)


'''

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
y = step_function(x)
plt.plot(x, y)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
'''

A = np.array([[1, 2], [3, 4]])
A.shape
B = np.array([[5, 6], [7, 8]])
B.shape
print(np.dot(A, B))


#x = np.array([-1.0, 1.0, 2.0])
#print(sigmoid(x))

#y = x > 0
#print(y.astype(np.int))


'''
w = np.array([0.5, 0.5])
b = -0.7


print(w*x)
print(np.sum(w*x)+b)
'''

'''
AND 게이트 구현

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w2 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


print(AND(0, 0))
print(AND(1, 1))
'''


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


x = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)  # (2,3)
print(x.shape)
print(B1.shape)

A1 = np.dot(x, W1) + B1
print(A1)
Z1 = sigmoid(A1)
print(Z1)

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape)