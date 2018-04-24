from torch import save, load, topk, eq, squeeze, sum, FloatTensor
from torch.autograd.variable import Variable
from numpy import zeros
from os import path


class AudioTrainer:
    def __init__(self, configs, model, train_loader, loss_fn, optimizer, logger, load_path=None, save_path=None,
                 test_loader=None):
        self.configs = configs
        self.model = model
        self.train_data = train_loader
        self.loss = loss_fn
        self.optimizer = optimizer
        self.logger = logger

        self.load_path = load_path
        self.save_path = save_path

        self.test_data = test_loader

    def train(self):
        for epoch in range(0, self.configs.epochs):

            accuracy_sum, loss_sum, n_iter = 0, 0, 0
            while n_iter < self.configs.iterations:
                try:
                    for i, datum in enumerate(self.train_data):

                        outputs, labels, loss = self.train_step(datum)
                        if outputs.shape[0] == 1:
                            continue

                        current_accuracy = self.calculate_accuracy(outputs, labels)
                        accuracy_sum += current_accuracy
                        loss_sum += loss.data[0]
                        n_iter += 1

                        if n_iter % self.configs.log_interval == 0:
                            self.log_batch_results(current_accuracy, loss.data[0], n_iter, epoch)
                            print("{}.{} complete....".format(epoch, n_iter))
                        if n_iter >= self.configs.iterations:
                            break

                except TypeError:
                    print("Error on Epoch {}...".format(epoch))

            test_acc = self.test_model()
            self.log_epoch_results(accuracy_sum, loss_sum, test_acc, n_iter, epoch)

            self.save_model()
            print("Epoch {} complete.".format(epoch))
            self.logger.export()
        self.logger.close()

    def test_model(self):
        accuracy_sum, n_iter = 0, 0
        while n_iter < self.configs.test_iterations:
            try:
                for i, datum in enumerate(self.test_data):
                    outputs, labels = self.feed_forward(datum)
                    if outputs.shape[0] == 1:
                            continue
                    else:
                        accuracy_sum += self.calculate_accuracy(outputs, labels)
                        n_iter += 1
            except ValueError:
                print("Error on iter {}...".format(n_iter))
        return accuracy_sum / (n_iter * self.configs.batch_size)

    def feed_forward(self, datum):
        # Load inputs and make them cuda accessible
        inputs, labels = self.convert_data(datum)

        # Skip this iteration if the batch is incomplete
        if not inputs.shape[0] == labels.shape[0] == self.configs.batch_size:
            out = zeros((1, 1))
            return out, out

        # Zero out grads, go forward
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        return outputs, labels

    def train_step(self, datum):
        outputs, labels = self.feed_forward(datum)

        if outputs.shape[0] == 1:
            return outputs, labels, labels

        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return outputs, labels, loss

    @staticmethod
    def convert_data(datum):
        inputs, labels = datum
        inputs = inputs.type(FloatTensor)
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

        return inputs, labels

    @staticmethod
    def calculate_accuracy(outputs, labels):
        _, max_index = topk(outputs, k=1)
        accuracy = sum(eq(squeeze(max_index), labels)).data[0]

        return accuracy

    def log_epoch_results(self, accuracy_sum, loss_sum, test_acc, iter, epoch):
        self.logger.add_data_point(self.configs.avg_acc_train, accuracy_sum/(iter * self.configs.batch_size), epoch)
        self.logger.add_data_point(self.configs.avg_loss, loss_sum/iter, epoch)
        self.logger.add_data_point(self.configs.avg_acc_test, test_acc, epoch)

    def log_batch_results(self, accuracy, loss, iter, epoch):
        n_iter = (epoch * self.configs.iterations) + iter
        self.logger.add_data_point(self.configs.inst_acc, accuracy/self.configs.batch_size, n_iter)
        self.logger.add_data_point(self.configs.inst_loss, loss, n_iter)

    def load_model(self):
        """
        Call this before training to load previous version of model
        :return: None
        """
        if self.load_path is not None:
            if path.isfile(self.load_path):
                self.model.load_state_dict(load(self.load_path))
            else:
                print("Load file not found, starting fresh....\n")

    def save_model(self):
        if self.save_path is not None:
            save(self.model.state_dict(), self.save_path)