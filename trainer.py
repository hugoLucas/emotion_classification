from torch import save, load, topk, eq, squeeze, sum
from torch.autograd.variable import Variable
from os.path import isfile


class AudioTrainer:
    def __init__(self, configs, audio_model, audio_data, audio_loss, audio_optimizer, save_path=None):
        self.configs = configs
        self.model = audio_model
        self.data = audio_data
        self.loss = audio_loss
        self.optimizer = audio_optimizer

        self.save_path = save_path

    def train(self):
        historical_accuracy_data, historical_loss_data = {}, {}

        for epoch in self.configs.epochs:

            accuracy_data, loss_data, iterations = [], [], 0
            for i, data in enumerate(self.data):

                # Load inputs and make them cuda accessible
                inputs, labels = data
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

                # Add data to current iterations results
                accuracy_data.append(accuracy)
                loss_data.append(loss)
                iterations += 1

            # Log results
            historical_accuracy_data[epoch] = sum(accuracy_data)/(iterations * self.configs.batch_size)
            historical_loss_data[epoch] = sum(loss_data)/(iterations * self.configs.batch_size)

            # Save model
            if self.save_path is not None:
                save(self.model.state_dict(), self.save_path)

        return historical_accuracy_data, historical_loss_data

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
