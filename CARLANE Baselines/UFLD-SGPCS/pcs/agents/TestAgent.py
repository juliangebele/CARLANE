import torch
from pcs.models.model import Net
from pcs.models.head import Classifier, CosineClassifier
from pcs.runtime.input_modules import InputModule
from pcs.runtime.frame_processor import FrameProcessor
from pcs.runtime.output_modules import VisualOut, JsonOut, TestOut, ProdOut


class TestAgent(object):
    """
    Supports multiple input modules
        - images: image dataset from a text file containing a list of paths to images
        - video: video input
        - camera: e.g. webcam
        - screencap: stream

    Supports multiple output modules
        - video: generates an output video; same as demo.py
        - test: compares result with labels; same as test.py
        - json: generates a json file as output
        - prod - will probably only log results, in future you will put your production code here
    """

    def __init__(self, config):
        self.config = config
        self.input_mode = self.config.input_params.input_mode
        self.output_mode = self.config.output_params.output_mode
        self.measure_time = self.config.test_params.measure_time

    def load_model(self):
        """
        Setup and load neural network
        """
        trained_model = self.config.test_params.trained_model
        backbone = self.config.model_params.backbone.split("-")[1]
        griding_num = self.config.model_params.griding_num
        h_samples = self.config.data_params.h_samples
        num_lanes = self.config.model_params.num_lanes
        cls_dim = (griding_num + 1, len(h_samples), num_lanes)
        temp = self.config.model_params.cls_temp

        assert backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

        torch.backends.cudnn.benchmark = True
        model = Net(pretrained=False, backbone=backbone, cls_dim=cls_dim, use_aux=False).cuda()

        # Load PCS_UFLD model
        if trained_model.endswith(".tar"):
            cls_head = CosineClassifier(cls_dim=cls_dim, temp=temp).cuda()
            trained_model_state_dict = torch.load(trained_model, map_location="cpu")

            model_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict["model_state_dict"].items()}
            model.load_state_dict(model_state_dict, strict=False)

            if "cls_state_dict" in trained_model_state_dict:
                cls_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict["cls_state_dict"].items()}
                cls_head.load_state_dict(cls_state_dict)

            print(f"Model successfully loaded from '{trained_model}', val acc = {trained_model_state_dict['val_acc']}\n")

        # Load UFLD model
        elif trained_model.endswith(".pth"):
            cls_head = Classifier(cls_dim=cls_dim).cuda()
            if 'DANN' in trained_model or 'ADDA' in trained_model or 'SGADA' in trained_model:
                trained_model_state_dict = torch.load(trained_model, map_location="cpu")["encoder"]
            else:
                trained_model_state_dict = torch.load(trained_model, map_location="cpu")["model"]
            trained_model_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict.items()}

            model_state_dict = {key: value for key, value in trained_model_state_dict.items() if "model" in key or "aux" in key or "pool" in key}
            model.load_state_dict(model_state_dict, strict=False)

            if 'DANN' in trained_model or 'ADDA' in trained_model or 'SGADA' in trained_model:
                trained_model_state_dict = torch.load(trained_model, map_location="cpu")["classifier"]
                trained_model_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict.items()}
            cls_state_dict = {key: value for key, value in trained_model_state_dict.items() if "cls" in key}
            cls_head.load_state_dict(cls_state_dict, strict=False)

            print(f"Model successfully loaded from '{trained_model}'\n")
        else:
            raise NotImplementedError("Model not supported or not properly loaded")

        model.eval()
        cls_head.eval()

        return model, cls_head

    def setup_input(self, process_frames):
        """
        Setup data input (where the frames come from)

        Args:
            process_frames: function taking list of frames and a corresponding list of filenames
        """
        input_module = InputModule(self.config)
        if self.input_mode == 'images':
            input_module.input_images(process_frames)
        elif self.input_mode == 'video':
            input_module.input_video(process_frames)
        elif self.input_mode == 'camera':
            input_module.input_camera(process_frames)
        elif self.input_mode == 'screencap':
            input_module.input_screencap(process_frames)
        else:
            print(self.input_mode)
            raise NotImplemented('unknown/unsupported input_mode')

    def setup_out_method(self):
        """
        Setup the output method

        Returns: method/function reference to a function taking

        * a list of predictions
        * a list of corresponding filenames (if available)
        * a list of source_frames (if available)

        """
        methods = []
        for output_mode in self.output_mode:
            if output_mode == 'video':
                video_out = VisualOut(self.config)
                methods.append((video_out.out, lambda: None))
            elif output_mode == 'test':
                test_out = TestOut(self.config)
                methods.append((test_out.out, test_out.post))
            elif output_mode == 'json':
                json_out = JsonOut(self.config)
                methods.append((json_out.out, lambda: None))
            elif output_mode == 'prod':
                prod_out = ProdOut()
                methods.append((prod_out.out, prod_out.post))
            else:
                print(output_mode)
                raise NotImplemented('unknown/unsupported output_mode')

        def out_method(*args, **kwargs):
            """
            Call all out_methods and pass all arguments to them
            """
            for method in methods:
                method[0](*args, **kwargs)

        def post_method(*args, **kwargs):
            """
            Call all post_methods and pass all arguments to them
            """
            for method in methods:
                method[1](*args, **kwargs)

        return out_method, post_method

    def run(self):
        model, cls_head = self.load_model()
        out_method, post_method = self.setup_out_method()
        measure_time = self.measure_time
        frame_processor = FrameProcessor(model, cls_head, out_method, measure_time)
        self.setup_input(frame_processor.process_frames)
        post_method()  # called when input method is finished (post processing)
