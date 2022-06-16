import cv2
import numpy as np
from tqdm import tqdm
from mss import mss
from PIL import Image
from itertools import count
from torchvision import transforms
from pcs.data.datautils import create_dataset_test, create_loader


class InputModule(object):
    def __init__(self, config):
        self.config = config

        self.img_height = config.data_params.image_height
        self.img_width = config.data_params.image_width
        self.image_height_net = self.config.data_params.image_height_net
        self.image_width_net = self.config.data_params.image_width_net
        self.img_transform = transforms.Compose([transforms.Resize((self.image_height_net, self.image_width_net)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    def input_images(self, process_frames):
        """
        Load images frame by frame from text_txt file and passes them to process_frame

        Args:
            process_frames: function taking a list of preprocessed frames, file paths and source frames
        """
        data_root = self.config.data_params.data_root

        images_input_file = self.config.input_params.images_input_file
        batch_size = self.config.model_params.batch_size
        num_workers = self.config.data_params.num_workers

        dataset_test = create_dataset_test(data_root, images_input_file, image_height=self.image_height_net, image_width=self.image_width_net)
        loader = create_loader(dataset_test, batch_size, num_workers, is_train=False)

        for i, data in enumerate(tqdm(loader)):
            imgs, names = data
            process_frames(imgs, names, None)

    def input_screencap(self, process_frames):
        """
        Record from screen, batch size is always 1

        You have to manually specify the position and size of your target window here.
        If your information is wrong (out of screen) you'll get a cryptic exception!
        Make sure your config resolution matches your settings here.

        used non-basic cfg options: screencap_enable_image_forwarding

        Args:
            process_frames: function taking a list of preprocessed frames, file paths and source frames
        """

        sct = mss()
        resize = False

        screencap_enable_image_forwarding = self.config.input_params.screencap_enable_image_forwarding
        mon = {
           'left': self.config.input_params.screencap_recording_area[0],
           'top': self.config.input_params.screencap_recording_area[1],
           'width': self.config.input_params.screencap_recording_area[2],
           'height': self.config.input_params.screencap_recording_area[3]
        }

        for i in count():
            screenshot = sct.grab(mon)
            image = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)

            # unsqueeze: adds one dimension to tensor array (to be similar to loading multiple images)
            frame = self.img_transform(image).unsqueeze(0)

            if screencap_enable_image_forwarding:
                image = np.array(image)
                # resize recorded frames if resolution is different from self.img_height / self.img_width
                if i == 0 and (image.shape[0] != self.img_height or image.shape[1] != self.img_width):
                    resize = True
                if resize:
                    image = cv2.resize(image, (self.img_width, self.img_height))

                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                process_frames(frame, None, [image])
            else:
                process_frames(frame, None, None)

    def input_video(self, process_frames):
        """
        Read a video file or camera stream. batch size is always 1

        Args:
            process_frames: function taking a list of preprocessed frames, file paths and source frames
        """

        # input_file: video file (path; string) or camera index (integer)
        # names_file: list with file paths to the frames of the video; if names_file and frames (jpg's) are available the input images module can also be used
        # camera_number: opencv camera index
        input_file = self.config.input_params.video_input_file

        if isinstance(input_file, int):
            input_file = self.config.input_params.camera_input_cam_number
        names_file = self.config.test_params.test_txt

        if names_file:
            with open(names_file, 'r') as file:
                image_paths = file.read().splitlines()
        # else:
        #     print('no names_file specified, some functions (output modules) might not work as they require names!')

        vid = cv2.VideoCapture(input_file)
        resize = False

        # scale / resize
        if isinstance(input_file, int):  # input is camera
            vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_width)
            vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_height)
            print(f'Input is a camera. Make sure your camera is capable of your selected resolution {self.img_width}x{self.img_height}')

        for i in count():
            success, image = vid.read()
            if not success:
                break

            if i == 0:
                if image.shape[0] != self.img_height or image.shape[1] != self.img_width:
                    resize = True
                    print('Your video file does not match the specified image size, resizing frames will probably impact performance.')

            if resize:
                image = cv2.resize(image, (self.img_width, self.img_height))

            frame = Image.fromarray(image)
            frame = self.img_transform(frame).unsqueeze(0)

            process_frames(frame, [image_paths[i]] if names_file else None, [image])

    def input_camera(self, process_frames):
        """
        Camera input wrapper for input_video()

        Args:
            process_frames: function taking a list of preprocessed frames, file paths and source frames
        """

        self.input_video(process_frames)
