from torch import save, load, topk, eq, squeeze, sum, FloatTensor
from torch.autograd.variable import Variable
from os import path


class AudioTrainer:
    def __init__(self, configs, model, loader, loss_fn, optimizer, logger, load_path=None, save_path=None):
        self.configs = configs
        self.model = model
        self.data = loader
        self.loss = loss_fn
        self.optimizer = optimizer
        self.logger = logger

        self.load_path = load_path
        if self.load_path is not None:
            self.save_path = self.load_path if save_path is None else save_path

    def train(self):
        for epoch in range(0, self.configs.epochs):

            accuracy_sum, loss_sum, n_iter = 0, 0, 0
            while n_iter < self.configs.iterations:
                # try:
                for i, datum in enumerate(self.data):

                    # Load inputs and make them cuda accessible
                    inputs, labels = datum
                    inputs = inputs.type(FloatTensor)
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

                    # Skip this iteration if the batch is incomplete
                    if not inputs.shape[0] == labels.shape[0] == self.configs.batch_size:
                        continue

                    # Zero out grads, go forward
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)

                    # Calculate loss and propagate backwards
                    loss = self.loss(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    # Calculate accuracy of predictions
                    _, max_index = topk(outputs, k=1)
                    accuracy = sum(eq(squeeze(max_index), labels)).data[0]

                    # Track epoch results
                    accuracy_sum += accuracy
                    loss_sum += loss.data[0]
                    n_iter += 1

                    # Log iteration results
                    if n_iter % self.configs.log_interval == 0:
                        self.log_batch_results(accuracy, loss.data[0], n_iter, epoch)
                        print("{}.{} complete....".format(epoch, n_iter))

                    if n_iter >= self.configs.iterations:
                        break

                # except TypeError:
                #     print("Error on Epoch {}...".format(epoch))
                #     break

            self.log_epoch_results(accuracy_sum, loss_sum, n_iter, epoch)

            # Save model
            if self.save_path is not None:
                save(self.model.state_dict(), self.save_path)

            print("Epoch {} complete.".format(epoch))

            self.logger.export()

        self.logger.close()

    def log_epoch_results(self, accuracy_sum, loss_sum, iter, epoch):
        self.logger.add_data_point(self.configs.avg_acc, accuracy_sum/(iter * self.configs.batch_size), epoch)
        self.logger.add_data_point(self.configs.avg_loss, loss_sum/iter, epoch)

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
                raise ValueError("Cannot find model save file: " + self.load_path)
