#!/usr/bin/env python3

import sys
import torch.utils.data
from PIL import Image
from data.base_dataset import get_params, get_transform

from itertools import islice

from PySide2.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout,
                               QWidget, QFileDialog)
from PySide2.QtCore import Slot, Qt

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from data.aligned_dataset import AlignedDataset


class LatentExplorer(QWidget):
    def __init__(self, model, opt):
        QWidget.__init__(self)

        self.dialog = QFileDialog(self)
        self.dialog.setFileMode(QFileDialog.ExistingFile)
        self.dialog.setNameFilter("Images (*.png *.jpg)")

        # BicycleGAN
        self.model = model
        self.opt = opt
        self.input_path = None
        self.z = None

        # Buttons
        self.button_load_image = QPushButton("Load image")
        self.button_generate = QPushButton("Generate")
        self.button_random_z = QPushButton("Random z")

        # Text
        self.text_input_path = QLabel(self.input_path)
        self.text_input_path.setAlignment(Qt.AlignCenter)
        self.text_z = QLabel(self.z)
        self.text_z.setAlignment(Qt.AlignCenter)

        # Layout
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.text_input_path)
        self.layout.addWidget(self.text_z)
        self.layout.addWidget(self.button_load_image)
        self.layout.addWidget(self.button_random_z)
        self.layout.addWidget(self.button_generate)
        self.setLayout(self.layout)

        # Connecting the signals
        self.button_load_image.clicked.connect(self.choose_image)
        self.button_random_z.clicked.connect(self.generate_random_z)
        self.button_generate.clicked.connect(self.generate)

        # Hacks for testing
        self.set_input_path('/home/ploth/projects/pair-images/files/1fd35396a5437cf4397fdfc5dd4b0973c3865f5d_contour.jpg')
        self.load_input_image()
        self.generate_random_z()

    def generate(self):
        self.model.set_input(self.data)
        self.test()

    def set_input_path(self, path):
        self.input_path = path
        self.text_input_path.setText(self.input_path)

    def choose_image(self):
        # Get path from file chooser
        self.dialog.exec_()
        path = self.dialog.selectedFiles()
        self.set_input_path(path)
        self.load_input_image()

    def load_input_image(self):
        # Load image
        self.image = Image.open(self.input_path).convert('RGB')
        # Convert image to tensor
        self.tensor_from_image()
        # Prepare GAN input
        self.data_from_tensor()

    def tensor_from_image(self):
        parameters = get_params(self.opt, self.image.size)
        #TODO Support for BtoA only!
        transform = get_transform(
            self.opt, parameters, grayscale=(self.opt.input_nc == 1))
        tensor = transform(self.image)
        # Simulate batch of size 1
        self.input_tensor = torch.unsqueeze(transform(self.image), 0)

    def data_from_tensor(self):
        self.data = {
            'A': self.input_tensor,
            'B': self.input_tensor,
            'A_paths': self.input_path,
            'B_paths': self.input_path
        }

    def generate_random_z(self):
        self.z = self.model.get_z_random(1, self.opt.nz)
        self.text_z.setText(self.z.__str__())

    def test(self):
        _, self.fake_B, _ = self.model.test(self.z, encode=False)
        self.np_image = util.tensor2im(self.fake_B)
        util.save_image(self.np_image, './test.png')


if __name__ == "__main__":
    # Load options
    opt = TestOptions().parse()
    # Test code only supports num_threads=1
    opt.num_threads = 1
    # Test code only supports batch_size=1
    opt.batch_size = 1
    # No shuffle
    opt.serial_batches = True

    # Create and setup model
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    app = QApplication(sys.argv)

    widget = LatentExplorer(model, opt)
    widget.resize(1920, 1080)
    widget.show()

    sys.exit(app.exec_())
