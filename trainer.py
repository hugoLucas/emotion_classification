from torch import save, load, topk, eq, squeeze, sum, FloatTensor
from torch.autograd.variable import Variable
from numpy import zeros
from os import path


def convert_data(datum):
    """
    Converts data loaded from Librosa into a format that PyTorch can handle.

    :param datum:   output of Dataset.__getitem___
    :return: output of Dataset.__getitem___ in valid format
    """
    inputs, labels = datum
    inputs = inputs.type(FloatTensor)
    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

    return inputs, labels


def calculate_accuracy(outputs, labels):
    """
    Calculates the number of correct predictions made by a model in a single iteration. Gets index of the largest value
    in the model output and compares it to the correct label.

    :param outputs:     an [N, 8] array containing the model output for a given set of input audio files
    :param labels:      an [N] array containing the correct labels for a given set of input audio files
    :return:
    """
    _, max_index = topk(outputs, k=1)
    accuracy = sum(eq(squeeze(max_index), labels)).data[0]

    return accuracy


class AudioTrainer:

    """
    Coordinates the training of a PyTorch model.
    """

    def __init__(self, configs, model, train_loader, loss_fn, optimizer, logger, load_path=None, save_path=None,
                 test_loader=None):
        """
        :param configs:         a Bunch object of this runs's configuration file
        :param model:           the model you wish to train, must extend torch.nn.Module
        :param train_loader:    the training data, must extend torch.utils.data.Dataset
        :param loss_fn:         the loss function to use in training, must extend torch.nn.modules.loss._Loss
        :param optimizer:       the optimization algorithm to use in training, must extend torch.optim.Optimizer
        :param logger:          a wrapper for TensorboardX
        :param load_path:       path to .pt file to load before training begins
        :param save_path:       path to directory where you want to save .pt file of your model
        :param test_loader:     the test data, must extend torch.utils.data.Dataset
        """
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
        """
        Trains self.model for the amount of epochs and iterations specified in configuration file. Periodically logs
        results of training and also tests model's current performance on the test set.
        :return: None
        """
        for epoch in range(0, self.configs.epochs):
            accuracy_sum, loss_sum, n_iter = 0, 0, 0
            while n_iter < self.configs.iterations:
                try:
                    for i, datum in enumerate(self.train_data):
                        outputs, labels, loss = self.train_step(datum)
                        if outputs.shape[0] == 1:
                            continue
                        current_accuracy = calculate_accuracy(outputs, labels)
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
        """
        Tests the models current performance on the test set, i.e. data that it is not being trained on.
        :return: None
        """
        accuracy_sum, n_iter = 0, 0
        while n_iter < self.configs.test_iterations:
            try:
                for i, datum in enumerate(self.test_data):
                    outputs, labels = self.feed_forward(datum)
                    if outputs.shape[0] == 1:
                            continue
                    accuracy_sum += calculate_accuracy(outputs, labels)
                    n_iter += 1

                    if n_iter >= self.configs.test_iterations:
                        break
            except ValueError:
                print("Error on iter {}...".format(n_iter))
        return accuracy_sum / (n_iter * self.configs.batch_size)

    def train_step(self, datum):
        """
        Given the output of Dataset object, method will feed data to self.model and use self.loss and self.optimzer
        to perform backpropogation.

        :param datum: result of Dataset.__getitem__
        :return: the model output, the labels for the batch, value of loss function for the iteration
        """
        outputs, labels = self.feed_forward(datum)

        if outputs.shape[0] == 1:
            return outputs, labels, labels

        loss = self.loss(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return outputs, labels, loss

    def feed_forward(self, datum):
        """
        Given the output of Dataset object, this method will feed said data into self.model and return the results as
        well as the labels contained in the input.

        :param datum:   result of Dataset.__getitem__
        :return:        model output, data labels
        """
        # Load inputs and make them cuda accessible
        inputs, labels = convert_data(datum)

        # Skip this iteration if the batch is incomplete
        if not inputs.shape[0] == labels.shape[0] == self.configs.batch_size:
            out = zeros((1, 1))
            return out, out

        # Zero out grads, go forward
        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        return outputs, labels

    def log_epoch_results(self, accuracy_sum, loss_sum, test_acc, n_iter, epoch):
        """
        Logs the results for a training epoch.

        :param accuracy_sum:    the sum total of the number of examples the model has predicted correctly for an epoch
        :param loss_sum:        the sum total of the value of the loss function for the epoch
        :param test_acc:        the accuracy on the test set for an epoch
        :param n_iter:          the number of iterations the epoch took to complete
        :param epoch:           the number of epochs completed
        :return:                None
        """
        self.logger.add_data_point(self.configs.avg_acc_train, accuracy_sum/(n_iter * self.configs.batch_size), epoch)
        self.logger.add_data_point(self.configs.avg_loss, loss_sum/n_iter, epoch)
        self.logger.add_data_point(self.configs.avg_acc_test, test_acc, epoch)

    def log_batch_results(self, accuracy, loss, n_iter, epoch):
        """
        Logs the results of a single iteration.

        :param accuracy:    the number of correct predictions made on this iteration
        :param loss:        the value of the loss function on this iteration
        :param n_iter:      the value of the iteration this data was taken from
        :param epoch:       the value of the epoch this iteration took place in
        :return:            None
        """
        n_iter = (epoch * self.configs.iterations) + n_iter
        self.logger.add_data_point(self.configs.inst_acc, accuracy/self.configs.batch_size, n_iter)
        self.logger.add_data_point(self.configs.inst_loss, loss, n_iter)

    def load_model(self):
        """
        Call this before training to load previous version of model.

        :return: None
        """
        if self.load_path is not None:
            if path.isfile(self.load_path):
                self.model.load_state_dict(load(self.load_path))
            else:
                print("Load file not found, starting fresh....\n")

    def save_model(self):
        """
        Saves model is a valid save path was provided.

        :return: None
        """
        if self.save_path is not None:
            save(self.model.state_dict(), self.save_path)
        else:
            print("Invalid save path - stop training and set valid save path")
