#!/usr/bin/env python3

import sys
import datetime
import torch.utils.data

from PIL import Image
from pathlib import Path
from data.base_dataset import get_params, get_transform

from PySide2.QtWidgets import (QApplication, QLabel, QPushButton, QVBoxLayout,
                               QWidget, QFileDialog, QBoxLayout, QGridLayout,
                               QSlider, QLineEdit, QCheckBox)
from PySide2.QtGui import QPainter, QColor, QFont, QImage, QPixmap, QPen
from PySide2.QtCore import Slot, Qt, QPoint, QRect

from options.test_options import TestOptions
from models import create_model
from util import util
from data.aligned_dataset import AlignedDataset


class DrawArea(QWidget):
    def __init__(self, opt, generator_input_path):
        super().__init__()

        # Options
        self.opt = opt

        # Variables
        self.image_loaded = False

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
        self.color = Qt.black
        self.pen_size = 1
        self.lastPoint = QPoint()

    def toggle_pen_eraser(self):
        if self.color is Qt.black:
            self.color = Qt.white
            self.pen_size = 10
        else:
            self.color = Qt.black
            self.pen_size = 1

    def clear_page(self):
        self.pixmap.fill(Qt.white)
        self.export_pixmap(self.generator_input_path)
        self.repaint()

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
            painter.setPen(QPen(self.color, self.pen_size, Qt.SolidLine))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.export_pixmap(self.generator_input_path)
        if event.button == Qt.LeftButton:
            self.drawing = False

    def __process_new_pixmap(self, path):
        self.pixmap = QPixmap(path)
        self.image_loaded = True
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

    def set_z(self, values):
        device = torch.device('cuda', self.opt.gpu_ids[0])
        self.z = torch.tensor([values], dtype=torch.float, device=device)

    def generate_random_z(self):
        self.z = self.model.get_z_random(1, self.opt.nz)
        return self.z

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

        # Variables
        self.opt = opt
        self.z = [0] * self.opt.nz

        # Check boxes
        self.check_box_live_update = QCheckBox("Live update")
        self.check_box_live_update.setChecked(True)

        # Buttons
        self.button_load_image = QPushButton("Load image")
        self.button_generate = QPushButton("Generate")
        self.button_reset_z = QPushButton("Reset z")
        self.button_generate_random_z = QPushButton("Generate random z")
        self.button_generate_random_sample = QPushButton(
            "Generate random sample")
        self.button_generate_multiple_random_samples = QPushButton(
            "Generate multiple random samples")
        self.button_toggle_pen_eraser = QPushButton("Pen / Eraser")
        self.button_clear_page = QPushButton("Clear page")
        self.button_auto_explore = QPushButton("Auto explore")

        # Sliders
        self.slider_max = 3
        self.slider_factor = 10000
        self.sliders = [QSlider(Qt.Horizontal) for i in range(0, self.opt.nz)]
        for slider in self.sliders:
            slider.setMinimum(-self.slider_max * self.slider_factor)
            slider.setMaximum(self.slider_max * self.slider_factor)
            slider.setSingleStep(self.slider_factor / 100)
            slider.setPageStep(self.slider_factor / 10)
            slider.setValue(0)

        # Text boxes
        self.text_boxes = [QLineEdit() for i in range(0, self.opt.nz)]
        for i, box in enumerate(self.text_boxes):
            box.setText(f'{self.sliders[i].value():.4f}')

        # Images
        self.draw_area = DrawArea(self.opt, self.generator_input_path)
        self.generator = Generator(self.opt, self.generator_input_path,
                                   self.generator_output_path, model)

        # Layout
        self.layout = QGridLayout()
        self.layout.addWidget(self.draw_area)
        self.layout.addWidget(self.button_toggle_pen_eraser)
        self.layout.addWidget(self.button_clear_page)
        for box, slider in zip(self.text_boxes, self.sliders):
            self.layout.addWidget(box)
            self.layout.addWidget(slider)
        self.layout.addWidget(self.generator)
        self.layout.addWidget(self.check_box_live_update)
        self.layout.addWidget(self.button_load_image)
        self.layout.addWidget(self.button_reset_z)
        self.layout.addWidget(self.button_generate_random_z)
        self.layout.addWidget(self.button_generate)
        self.layout.addWidget(self.button_generate_random_sample)
        self.layout.addWidget(self.button_generate_multiple_random_samples)
        self.layout.addWidget(self.button_auto_explore)
        self.setLayout(self.layout)

        # Connecting the signals
        for slider in self.sliders:
            slider.sliderMoved.connect(self.sliders_edited)
        for box in self.text_boxes:
            box.textEdited.connect(self.text_boxes_edited)
        self.button_load_image.clicked.connect(self.draw_area.file_chooser)
        self.button_reset_z.clicked.connect(self.reset_z)
        self.button_generate_random_z.clicked.connect(self.generate_random_z)
        self.button_generate.clicked.connect(self.generate)
        self.button_generate_random_sample.clicked.connect(
            self.generate_random_sample)
        self.button_generate_multiple_random_samples.clicked.connect(
            self.generate_multiple_random_samples)
        self.button_toggle_pen_eraser.clicked.connect(self.draw_area.toggle_pen_eraser)
        self.button_clear_page.clicked.connect(self.draw_area.clear_page)
        self.button_auto_explore.clicked.connect(self.auto_explore)

    def auto_explore(self):
        self.reset_z()
        for slider in self.sliders:
            self.move_slider_from_min_to_max(slider)
            self.reset_z()

    def move_slider_from_min_to_max(self, slider):
        step = int(0.1 * self.slider_factor)
        maximum = slider.maximum() + step
        for value in range(slider.minimum(), maximum, step):
            slider.setValue(value)
            self.sliders_edited()

    def export_images(self):
        now = datetime.datetime.today()
        self.draw_area.export_pixmap(self.base_dir / 'latent_explorer' / str(str(now.isoformat()) + '_input.png'))
        self.generator.export_pixmap(self.base_dir / 'latent_explorer' / str(str(now.isoformat()) + '_output.png'))

    def generate(self):
        if self.draw_area.image_loaded:
            self.generator.generate()
            self.export_images()

    def reset_z(self):
        z = [0] * len(self.sliders)
        self.generator.set_z(z)
        self.update_input_widgets(z)
        self.generate()

    def generate_random_z(self):
        z = self.generator.generate_random_z()
        self.update_input_widgets(*(z.tolist()))

    def generate_multiple_random_samples(self):
        for i in range(0,100):
            self.generate_random_sample()

    def generate_random_sample(self):
        if self.draw_area.image_loaded:
            z = self.generator.generate_random_z()
            self.generate()
            self.update_input_widgets(*(z.tolist()))

    def update_input_widgets(self, values):
        for i, (box, slider) in enumerate(zip(self.text_boxes, self.sliders)):
            box.setText(f'{values[i]:.4f}')
            slider.setValue(values[i] * self.slider_factor)

    def sliders_edited(self):
        self.generator.set_z(
            [slider.value() / self.slider_factor for slider in self.sliders])
        for slider, box in zip(self.sliders, self.text_boxes):
            box.setText(f'{slider.value() / self.slider_factor:.4f}')

        if self.check_box_live_update.isChecked():
            self.generate()

    def text_boxes_edited(self):
        self.generator.set_z([float(box.text()) for box in self.text_boxes])
        for slider, box in zip(self.sliders, self.text_boxes):
            slider.setValue(float(box.text()) * self.slider_factor)

        if self.check_box_live_update.isChecked():
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
    widget.show()

    sys.exit(app.exec_())
