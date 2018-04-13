from tensorboardX import SummaryWriter


class AudioLogger:
    def __init__(self, destination):
        self.client = SummaryWriter()
        self.destination = destination

    def add_data_point(self, data_name, data_point, iter):
        self.client.add_scalar(data_name, data_point, iter)

    def export(self):
        self.client.export_scalars_to_json(self.destination)

    def close(self):
        self.client.close()
