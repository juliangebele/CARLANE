import os
import cv2
import numpy as np
from PIL import Image


class ImageSaver:
    """
    Stores incoming data in a Numpy ndarray and saves the array to disk once
    completely filled.
    """

    def __init__(self, filename, size, rows, cols, depth):
        """
        An array of shape (size, rows, cols, depth) is created to hold
        incoming images (this is the buffer). `filename` is where the buffer
        will be stored once full.
        """
        self.filename = filename
        self.size = size
        self.buffer = np.empty(shape=(size, rows, cols, depth), dtype=np.uint8)
        self.curves_list = []  # where to save images (left_curve, straight or right_curve)
        self.index = 0
        self.reset_count = 0  # how many times this object has been reset
        self.image_name = 0
        self.curve_counters = {
            'steep_left_curve/': 0,
            'left_curve/': 0,
            'straight/': 0,
            'right_curve/': 0,
            'steep_right_curve/': 0
        }

    def is_full(self):
        return self.index == self.size

    def reset(self):
        self.buffer = np.empty_like(self.buffer)
        self.curves_list = []
        self.index = 0
        self.reset_count += 1

    def save(self):
        try:
            for i, image_array in enumerate(self.buffer[:self.index + 1]):
                image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB), 'RGB')
                self.image_name = self.curve_counters[self.curves_list[i]]
                save_name = self.filename + self.curves_list[i] + f'{self.image_name:04d}'
                folder = os.path.dirname(save_name)

                if not os.path.isdir(folder):
                    os.makedirs(folder)

                image.save(save_name + '.jpg', 'JPEG')
                self.curve_counters[self.curves_list[i]] += 1
        except IndexError:
            print('WARNING: no full imagebuffer saved')

    @staticmethod
    def process_by_type(raw_img, name):
        """
        Converts the raw image to a more efficient processed version
        useful for training. The processing to be applied depends on the
        sensor name, passed as the second argument.
        """
        if name == 'CameraRGB':
            return raw_img
        elif name == 'CameraSemSeg':
            return raw_img[:, :, 2:3]  # only the red channel has information

    def add_image(self, img_bytes, name, curve_type):
        """
        Save the current buffer to disk and reset the current object
        if the buffer is full.
        """
        if self.is_full():
            self.save()
            self.reset()

        raw_image = np.frombuffer(img_bytes, dtype=np.uint8)
        raw_image = raw_image.reshape(self.buffer.shape[1], self.buffer.shape[2], -1)
        raw_image = self.process_by_type(raw_image[:, :, :3], name)
        self.buffer[self.index] = raw_image
        self.curves_list.append(curve_type)
        self.index += 1
