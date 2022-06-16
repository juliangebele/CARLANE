import os
import cv2

img_list = []


def load_images(image_dir):
    """
    Load images of a folder structure recursively.
    """
    for filename in os.listdir(image_dir):
        if filename.endswith('.jpg'):
            load_file = os.path.join(image_dir, filename)
            img_list.append(load_file)
        elif not filename.endswith('.png') and not filename.endswith('.json'):
            load_images(image_dir + filename + '/')
    return img_list


def create_video(root_dir, image_dir):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    vout = cv2.VideoWriter(os.path.join(root_dir, 'video.avi'), fourcc, 30.0, (1280, 720))

    images = load_images(image_dir)
    for image in images:
        path = os.path.join(root_dir, image)
        image = cv2.imread(path)
        cv2.imshow('video', image)
        cv2.waitKey(1)
        vout.write(image)


if __name__ == '__main__':
    create_video('./TuLane/data/', './TuLane/data/clips/0531/')
