import datetime
import tqdm
import os
import torch.utils.data

from PIL import Image
from pathlib import Path

from data.aligned_dataset import AlignedDataset
from data.base_dataset import get_params, get_transform
from models import create_model
from options.test_options import TestOptions
from util import util


def generate_random_samples(opt, model, input_path, number_of_samples=10):
    images = []
    for i in range(number_of_samples):
        image = generate(opt, model, input_path)

        # Save output image
        output_path = input_path.with_name(input_path.stem + '_' + str(i) +
                                           input_path.suffix)
        # Save image
        image_pil = Image.fromarray(image)
        image_pil.save(output_path)

        # Append image
        images.append(image)

    return images


def generate(opt, model, input_path):
    # Prepare model input
    data = prepare_model_input(str(input_path), opt)

    # Set model input
    model.set_input(data)

    # Run model and return image
    return run_model(opt, model)


def run_model(opt, model, z=None):
    if z is None:
        z = model.get_z_random(1, opt.nz)

    # Run model
    _, fake_B, _ = model.test(z, encode=False)

    # Return image
    return util.tensor2im(fake_B)


def prepare_model_input(path, opt):
    # Load image
    image = Image.open(path).convert('RGB')

    # Convert image to tensor
    input_tensor = tensor_from_image(image, opt)

    # Prepare GAN input
    return data_from_tensor(input_tensor, path)


def tensor_from_image(image, opt):
    parameters = get_params(opt, image.size)
    #TODO Support for BtoA only!
    transform = get_transform(opt, parameters, grayscale=(opt.input_nc == 1))
    tensor = transform(image)
    # Simulate batch of size 1
    return torch.unsqueeze(transform(image), 0)


def data_from_tensor(tensor, path):
    return {'A': tensor, 'B': tensor, 'A_paths': path, 'B_paths': path}


if __name__ == '__main__':
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

    input_folder = Path(opt.test_dir)
    input_images = list(input_folder.glob('**/*.png'))

    # Loop through imatest_folder_bakges
    #  for path in tqdm(input_images):
    for path in input_images:
        images = generate_random_samples(opt, model, path)
