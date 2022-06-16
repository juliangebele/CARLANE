import os


def rename_images(path):
    """
    Rename .jpg images in a folder.
    """
    image_name = 0
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            os.rename(os.path.join(path, file), f'{path}{image_name:04d}.jpg')
            image_name += 1


def reverse_order_and_rename_images(path):
    """
    Reverse the order of .jpg images in a folder and rename them.
    """
    image_name = len([name for name in os.listdir(path)]) - 1
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            os.rename(os.path.join(path, file), f'{path}{image_name:04d}.jpg')
            image_name -= 1


if __name__ == '__main__':
    rename_images('./MoLane/data/train/real/s_track/gray/a/')
    # reverse_order_and_rename_images('./MoLane/data/train/real/s_track/gray/a/')
