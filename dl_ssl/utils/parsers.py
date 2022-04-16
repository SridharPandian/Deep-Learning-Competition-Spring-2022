import argparse

class TrainParser(object):
    def __init__(self):
        # Initializing the parser
        self.parser = argparse.ArgumentParser()

        # Adding arguments
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument('-m', '--model', type=str)
        self.parser.add_argument('-n', '--encoder', default="resnet50", type=str)
        self.parser.add_argument('-b', '--batch_size', type=int)
        self.parser.add_argument('-e', '--epochs', default=200, type=int)
        self.parser.add_argument('-g', '--gpu_num', default=0, type=int)
        self.parser.add_argument('-l', '--lr', type=float)
        self.parser.add_argument('-i', '--img_size', default=224, type=int)
        self.parser.add_argument('-d', '--train_data_path', type=str)
        self.parser.add_argument('-r', '--run', type=int)
        self.parser.add_argument('-a', '--augment_imgs', action='store_true')
        self.parser.add_argument('--labelled', action='store_true')

    def parse(self):
        return self.parser.parse_args()