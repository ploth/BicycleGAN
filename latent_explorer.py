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
    def __init__(self, opt, generator_input_path):
        super().__init__()

        # Options
        self.opt = opt

        # Paths
        self.generator_input_path = generator_input_path

        # Display
        self.pixmap = QPixmap()
        self.setFixedSize(self.opt.load_size, self.opt.load_size)
        self.show()

        # File dialog
        self.dialog = QFileDialog(self)
        self.dialog.setFileMode(QFileDialog.ExistingFile)
        self.dialog.setNameFilter("Images (*.png *.jpg)")

        # Drawing utils
        self.drawing = False
        self.lastPoint = QPoint()

    def export_pixmap(self, path):
        self.pixmap.save(str(path), path.suffix[1:])

    def file_chooser(self):
        # Get path from file chooser
        self.dialog.exec_()
        path = self.dialog.selectedFiles()
        self.__process_new_pixmap(*path)
        self.export_pixmap(self.generator_input_path)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.pixmap)
            painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.export_pixmap(self.generator_input_path)
        if event.button == Qt.LeftButton:
            self.drawing = False

    def __process_new_pixmap(self, path):
        self.pixmap = QPixmap(path)
        self.repaint()


class Generator(QWidget):
    def __init__(self, opt, generator_input_path, generator_output_path,
                 model):
        super().__init__()

        # Options
        self.opt = opt

        # Paths
        self.generator_input_path = generator_input_path
        self.generator_output_path = generator_output_path
        self.input_path = None
        self.output_path = None

        # Display
        self.pixmap = QPixmap()
        self.setFixedSize(self.opt.load_size, self.opt.load_size)
        self.show()

        # BicycleGAN
        self.model = model
        self.opt = opt
        self.z = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.pixmap)

    def export_pixmap(self, path):
        self.pixmap.save(str(path), path.suffix[1:])

    def generate_random_z(self):
        self.z = self.model.get_z_random(1, self.opt.nz)

    def generate(self):
        # Prepare model input
        data = self.__prepare_model_input(
            str(self.generator_input_path), self.opt)

        # Set model input
        self.model.set_input(data)

        # Run model
        self.__run_model()

        # Display model's output
        self.__process_new_pixmap(str(self.generator_output_path))

    def __run_model(self):
        if self.z is None:
            self.generate_random_z()

        # Run model
        _, self.fake_B, _ = self.model.test(self.z, encode=False)

        # Convert model output
        image = util.tensor2im(self.fake_B)

        # Save output picture
        util.save_image(image, self.generator_output_path)

    def __process_new_pixmap(self, path):
        self.pixmap = QPixmap(path)
        self.repaint()

    def __prepare_model_input(self, path, opt):
        # Load image
        image = Image.open(path).convert('RGB')
        # Convert image to tensor
        input_tensor = self.__tensor_from_image(image, opt)
        # Prepare GAN input
        return self.__data_from_tensor(input_tensor, path)

    def __tensor_from_image(self, image, opt):
        parameters = get_params(opt, image.size)
        #TODO Support for BtoA only!
        transform = get_transform(
            opt, parameters, grayscale=(opt.input_nc == 1))
        tensor = transform(image)
        # Simulate batch of size 1
        return torch.unsqueeze(transform(image), 0)

    def __data_from_tensor(self, tensor, path):
        return {'A': tensor, 'B': tensor, 'A_paths': path, 'B_paths': path}


class LatentExplorer(QWidget):
    def __init__(self, model, opt):
        QWidget.__init__(self)

        # Paths
        self.base_dir = Path(__file__).absolute().parents[0]
        self.generator_input_path = self.base_dir / 'latent_explorer' / 'input.png'
        self.generator_output_path = self.base_dir / 'latent_explorer' / 'output.png'

        # Buttons
        self.button_load_image = QPushButton("Load image")
        self.button_generate = QPushButton("Generate")
        self.button_generate_random_z = QPushButton("Generate random z")
        self.button_generate_random_sample = QPushButton(
            "Generate random sample")

        # Text
        self.z = None
        self.text_z = QLabel(self.z)
        self.text_z.setAlignment(Qt.AlignCenter)

        # Images
        self.draw_area = DrawArea(opt, self.generator_input_path)
        self.generator = Generator(opt, self.generator_input_path,
                                   self.generator_output_path, model)

        # Layout
        self.layout = QGridLayout()
        self.layout.addWidget(self.draw_area)
        self.layout.addWidget(self.text_z)
        self.layout.addWidget(self.generator)
        self.layout.addWidget(self.button_load_image)
        self.layout.addWidget(self.button_generate_random_z)
        self.layout.addWidget(self.button_generate)
        self.layout.addWidget(self.button_generate_random_sample)
        self.setLayout(self.layout)

        # Connecting the signals
        self.button_load_image.clicked.connect(self.draw_area.file_chooser)
        self.button_generate_random_z.clicked.connect(
            self.generator.generate_random_z)
        self.button_generate.clicked.connect(self.generate)
        self.button_generate_random_sample.clicked.connect(
            self.generate_random_sample)

    def export_images(self):
        pass

    def generate(self):
        self.generator.generate()
        self.export_images()

    def generate_random_sample(self):
        self.generator.generate_random_z()
        self.generate()


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
