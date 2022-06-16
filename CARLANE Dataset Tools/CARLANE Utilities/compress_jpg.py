import os
from PIL import Image


def compress_jpg(loading_dir):
    """
    Open .jpg images and save them to save space.
    """
    for filename in os.listdir(loading_dir):
        if filename.endswith('.jpg'):
            load_file = os.path.join(loading_dir, filename)
            print(load_file)
            img = Image.open(load_file)
            img.save(load_file, 'JPEG')
        elif not filename.endswith('.png'):
            compress_jpg(loading_dir + filename + '/')


if __name__ == '__main__':
    compress_jpg("./path/to/folder/")
