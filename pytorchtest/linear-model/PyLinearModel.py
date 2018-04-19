import numpy as np
import matplotlib.pyplot as plt


class PySimpleLinearModel:
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.w_list = []
        self.mse_list = []  # mean square error
        self.w = 1.0  # a random guess: random value

    # y = x * w
    def forward(self, x):
        return x * self.w

    # Loss function
    # (x*w -y)^2
    def loss(self, x, y):
        y_pred = self.forward(x)
        return (y_pred - y) * (y_pred - y)

    # compute gradient
    def gradient(self,x, y):  # d_loss/d_w
        return 2 * x * (x * self.w - y)

    # MSE
    def train_1(self):
        for w in np.arange(0.0, 4.1, 0.1):
            self.w=w
            print("w=", w)
            l_sum = 0
            for x_val, y_val in zip(self.x_data, self.y_data):
                y_pred_val = self.forward(x_val)
                l = self.loss(x_val, y_val)
                l_sum += l
                print("\t", x_val, y_val, y_pred_val, l)
            print("MSE=", l_sum / 3)
            self.w_list.append(w)
            self.mse_list.append(l_sum / 3)

    # GRADIENT DECENT
    def train_2(self,no_of_epoch=100):
        # Training loop
        for epoch in range(no_of_epoch):
            for x_val, y_val in zip(x_data, y_data):
                grad = self.gradient(x_val, y_val)
                self.w = self.w - 0.01 * grad
                print("\tgrad: ", x_val, y_val, round(grad, 2))
                l = self.loss(x_val, y_val)

            print("progress:", epoch, "w=", round(self.w, 2), "loss=", round(l, 2))

        # After training
        print("predict (after training)", "4 hours", self.forward(4))

    def show_chart(self):
        plt.plot(self.w_list, self.mse_list)
        plt.ylabel('Loss')
        plt.xlabel('w')
        plt.show()


x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
pslm = PySimpleLinearModel(x_data, y_data)
pslm.train_2(1000)
#pslm.show_chart()