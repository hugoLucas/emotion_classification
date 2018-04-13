from torch import save, load, topk, eq, squeeze, sum
from torch.autograd.variable import Variable
from utils.config_utils import sum_list
from os.path import isfile


class AudioTrainer:
    def __init__(self, configs, model, loader, loss_fn, optimizer, logger, save_path=None):
        self.configs = configs
        self.model = model
        self.data = loader
        self.loss = loss_fn
        self.optimizer = optimizer
        self.logger = logger

        self.save_path = save_path

    def train(self):
        for epoch in range(0, self.configs.epochs):

            accuracy_sum, loss_sum, iter = 0, 0, 0
            for i, datum in enumerate(self.data):

                # Load inputs and make them cuda accessible
                inputs, labels = datum
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
                iter += 1

                # Log iteration results
                self.log_batch_results(accuracy, loss.data[0], iter, epoch)

                if iter == self.configs.iterations:
                    break

            # Log epoch results
            if iter % self.configs.log_interval == 0:
                self.log_epoch_results(accuracy_sum, loss_sum, iter, epoch)

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
        if self.save_path is not None:
            if isfile(self.save_path):
                self.model.load_state_dict(load(self.save_path))
            else:
                raise ValueError("Cannot find model save file: " + self.save_path)
