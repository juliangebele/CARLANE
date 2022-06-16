import os
import cv2


def remove_images(image_dir):
    """
    Remove images of a folder structure recursively.
    """
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            if filename != '20.jpg':
                remove_file = os.path.join(image_dir, filename)
                os.remove(remove_file)
        else:
            remove_images(image_dir + filename + '/')


counter = 0


def count_images(image_dir):
    """
    Count images of a folder structure recursively.
    """

    global counter
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            counter += 1
        elif filename.endswith('.png'):
            counter += 1
        else:
            count_images(image_dir + filename + '/')


if __name__ == '__main__':
    count_images('./TuLane/data/clips/0313-2/')
    print(counter)
    remove_images('./TuLane/data/clips/0313-2/')
