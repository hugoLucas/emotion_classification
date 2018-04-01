import matplotlib.pyplot as plt


class AudioLogger:
    def __init__(self, acc_data, loss_data):
        self.acc_data = acc_data
        self.loss_data = loss_data

    def show_results(self):
        x_axis, acc, loss = self.get_data()

        plt.figure(1)
        plt.subplot(211)
        plt.plot(x_axis, acc, 'k')

        plt.subplot(212)
        plt.plot(x_axis, loss, 'k')

        plt.show()

    def get_data(self):
        keys = sorted(self.acc_data.keys())

        acc, loss = [], []
        for k in keys:
            acc.append(self.acc_data[k])
            loss.append(self.loss_data[k])
        return keys, acc, loss

