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
    def __init__(self, model, dataset, opt):
        QWidget.__init__(self)

        self.dialog = QFileDialog(self)
        self.dialog.setFileMode(QFileDialog.ExistingFile)
        self.dialog.setNameFilter("Images (*.png *.jpg)")

        self.model = model
        self.opt = opt
        self.aligned_dataset = AlignedDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.aligned_dataset, batch_size=1, shuffle=False, num_workers=1)
        self.dataset = dataset
        self.input_path = '/home/ploth/projects/pair-images/files/1fd35396a5437cf4397fdfc5dd4b0973c3865f5d_contour.jpg'

        self.button_load_image = QPushButton("Load image")
        self.button_generate_random_sample = QPushButton(
            "Generate random sample")
        self.z_sample = None
        self.text = QLabel(self.z_sample)
        self.text.setAlignment(Qt.AlignCenter)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.text)
        self.layout.addWidget(self.button_load_image)
        self.layout.addWidget(self.button_generate_random_sample)
        self.setLayout(self.layout)

        # Connecting the signal
        self.button_load_image.clicked.connect(self.load_input_image)
        self.button_generate_random_sample.clicked.connect(
            self.generate_random_sample)

    @Slot()
    def generate_random_sample(self):
        for i, data in enumerate(islice(self.dataloader, 1)):
            self.data = data
            print(data)
        print('now mine')
        self.tensor_from_image()
        #  self.data_from_tensor()
        print(self.data)
        self.model.set_input(self.data)
        self.generate_random_z()
        self.test()

    def load_input_image(self):
        # Get path from file chooser
        self.dialog.exec_()
        self.input_path = self.dialog.selectedFiles()
        print(self.input_path)
        # Load image
        # Convert image to tensor
        self.tensor_from_image()
        # Construct fake dataset object with A and B tensor an theirs paths
        # Save dataset element as member
        #  self.model.set_input(data)

    def tensor_from_image(self):
        image = Image.open(self.input_path).convert('RGB')
        parameters = get_params(self.opt, image.size)
        transform = get_transform(
            self.opt, parameters, grayscale=(self.opt.input_nc == 1))
        self.input_tensor = transform(image)

    def data_from_tensor(self):
        self.data = {
            'A': self.input_tensor,
            'B': self.input_tensor,
            'A_paths': self.input_path,
            'B_paths': self.input_path
        }

    def generate_random_z(self):
        self.z_sample = self.model.get_z_random(1, self.opt.nz)
        self.text.setText(self.z_sample.__str__())

    def test(self):
        _, self.fake_B, _ = self.model.test(self.z_sample, encode=False)
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

    # Create dataset
    dataset = create_dataset(opt)

    # Create and setup model
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    app = QApplication(sys.argv)

    widget = LatentExplorer(model, dataset, opt)
    widget.resize(1920, 1080)
    widget.show()

    sys.exit(app.exec_())
