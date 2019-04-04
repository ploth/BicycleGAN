#!/usr/bin/env python3

import sys
import datetime
import torch.utils.data

from PIL import Image
from pathlib import Path
from data.base_dataset import get_params, get_transform

from PySide2.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout,
                               QWidget, QFileDialog, QBoxLayout, QGridLayout)
from PySide2.QtGui import QPainter, QColor, QFont, QImage, QPixmap, QPen
from PySide2.QtCore import Slot, Qt, QPoint, QRect

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
from data.aligned_dataset import AlignedDataset


class DrawArea(QWidget):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.drawing = False
        self.lastPoint = QPoint()
        self.image = QPixmap()
        self.setFixedSize(self.opt.load_size, self.opt.load_size)
        self.show()

    def set_image(self, path):
        self.image = QPixmap(path)
        self.repaint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.image)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(Qt.red, 3, Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False


class LatentExplorer(QWidget):
    def __init__(self, model, opt):
        QWidget.__init__(self)

        self.dialog = QFileDialog(self)
        self.dialog.setFileMode(QFileDialog.ExistingFile)
        self.dialog.setNameFilter("Images (*.png *.jpg)")

        # BicycleGAN
        self.model = model
        self.opt = opt

        # Variables
        self.input_path = None
        self.output_path = None
        self.z = None
        self.drawing = False
        self.lastPoint = QPoint()
        self.base_dir = Path(__file__).absolute().parents[0]

        # Buttons
        self.button_load_image = QPushButton("Load image")
        self.button_generate = QPushButton("Generate")
        self.button_generate_random_z = QPushButton("Generate random z")
        self.button_generate_random_sample = QPushButton(
            "Generate random sample")

        # Text
        self.text_z = QLabel(self.z)
        self.text_z.setAlignment(Qt.AlignCenter)

        # Images
        self.image_draw_area = DrawArea(self.opt)
        self.image_generated = QLabel()

        # Layout
        self.layout = QGridLayout()
        self.layout.addWidget(self.image_draw_area)
        self.layout.addWidget(self.image_generated)
        self.layout.addWidget(self.text_z)
        self.layout.addWidget(self.button_load_image)
        self.layout.addWidget(self.button_generate_random_z)
        self.layout.addWidget(self.button_generate)
        self.layout.addWidget(self.button_generate_random_sample)
        self.setLayout(self.layout)

        # Connecting the signals
        self.button_load_image.clicked.connect(self.choose_image)
        self.button_generate_random_z.clicked.connect(self.generate_random_z)
        self.button_generate.clicked.connect(self.generate)
        self.button_generate_random_sample.clicked.connect(
            self.generate_random_sample)

    def generate(self):
        self.model.set_input(self.data)
        self.test()

    def set_input_path(self, path):
        self.input_path = path

    def set_output_path(self, path):
        self.output_path = path

    def choose_image(self):
        # Get path from file chooser
        self.dialog.exec_()
        path = self.dialog.selectedFiles()
        self.set_input_path(*path)
        self.load_input_image()

    def load_input_image(self):
        # Load to draw area
        self.image_draw_area.set_image(self.input_path)
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

    def generate_random_sample(self):
        self.generate_random_z()
        self.generate()

    def test(self):
        _, self.fake_B, _ = self.model.test(self.z, encode=False)
        self.np_image = util.tensor2im(self.fake_B)
        current_date = datetime.datetime.today()
        self.set_output_path(
            str(self.base_dir / 'latent_explorer' /
                str(current_date.isoformat())) + '.png')
        util.save_image(self.np_image, self.output_path)
        self.set_generated_image(self.output_path)

    def set_generated_image(self, path):
        self.image_generated.setPixmap(QPixmap(path))


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
