import numpy as np


class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim=1, lr=0.01):


        self.lr = lr

        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01

        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01

        self.b2 = np.zeros((1, output_dim))

    
    def relu(self, x):

        return np.maximum(0, x)
    
    def relu_grad(self, x):

        return (x>0).astype(float)
    
    def sigmod(self, x):
        return 1/(1+np.exp(-x))
    
    def forward(self, X):

        self.Z1 = np.dot(X, self.W1) + self.b1

        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.Z2) + self.b2 

        self.A2 = self.sigmod(self.Z2)

        return self.A2 
    
    def comput_loss(self, y_pred, y_true):

        m = y_true.shape[0]

        eps = 1e-8

        y_pred = np.clip(y_pred, eps, 1-eps)

        loss = -np.sum(
            y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred)
        )/m

        return loss 
    
    def backward(self, X, y_true):

        m = X.shape[0]


        # dZ2 = A2 - y
        dZ2 = self.A2 - y_true

        # dW2 = A1^T * dZ2 /m
        dW2 = np.dot(self.A1.T, dZ2) / m

        db2 = np.sum(dZ2, axis = 0, keepdims=True)/m

        # dA1 = dZ2 @ W2.T
        dA1 = np.dot(dZ2, self.W2.T)

        # ReLU 梯度
        dZ1 = dA1 * self.relu_grad(self.Z1)

        dW1 = np.dot(X.T, dZ1)/m 

        db1 = np.sum(dZ1, axis=0, keepdims=True) /m 

        self.W2 -= self.lr * dW2 
        self.b2 -= self.lr * db2 
        self.W1 -= self.lr * dW1 
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=100, print_every=100):

        for epoch in range(epochs):

            y_pred = self.forward(X)

            loss = self.comput_loss(y_pred, y)

            self.backward(X, y)

            if epoch%print_every == 0:
                print(f"epoch={epoch}, loss={loss:.6f}")

    def predict(self, X):

        probs = self.forward(X)

        return (probs>0.5).astype(int)



