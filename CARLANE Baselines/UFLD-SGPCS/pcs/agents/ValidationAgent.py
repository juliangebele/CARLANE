import os
import torch
from tqdm import tqdm
from pcs.data import datautils
from pcs.models.model import Net
from pcs.models.head import Classifier, CosineClassifier
from pcs.utils.metrics import update_metrics, reset_metrics, get_metric_dict


class ValidationAgent(object):
    """
    This class is used to validate on validation dataset with UFLD proposed metric.
    """

    def __init__(self, config):
        self.config = config
        self.out_dir = config.output_params.out_dir
        self.trained_model = self.config.test_params.trained_model
        self.use_aux = self.config.model_params.use_aux
        self.griding_num = self.config.model_params.griding_num
        self.num_lanes = self.config.model_params.num_lanes
        self.domain_map = {
            "source": self.config.data_params.source,
            "target": self.config.data_params.target,
        }
        self.val_loader_source = None
        self.val_loader_target = None
        self.test_loader = None
        self.metric_dict = get_metric_dict(self.griding_num, self.num_lanes)

        self.out_file = open(os.path.join(self.out_dir, 'validation_results.txt'), 'a')

    def load_dataset(self):
        data_root = self.config.data_params.data_root
        num_workers = self.config.data_params.num_workers
        griding_num = self.griding_num
        num_lanes = self.num_lanes
        use_aux = self.use_aux
        image_height = self.config.data_params.image_height_net
        image_width = self.config.data_params.image_width_net
        row_anchor = [x * image_height for x in self.config.data_params.h_samples]
        batch_size = self.config.model_params.batch_size

        source_val = self.config.data_params.source_val
        target_val = self.config.data_params.target_val
        target_test = self.config.input_params.images_input_file

        # Validation Dataset
        val_dataset_source = datautils.create_dataset_lbd(data_root,
                                                          source_val,
                                                          image_height=image_height,
                                                          image_width=image_width,
                                                          griding_num=griding_num,
                                                          row_anchor=row_anchor,
                                                          num_lanes=num_lanes,
                                                          use_aux=use_aux)

        self.val_loader_source = datautils.create_loader(val_dataset_source, batch_size, is_train=False, num_workers=num_workers)

        val_dataset_target = datautils.create_dataset_lbd(data_root,
                                                          target_val,
                                                          image_height=image_height,
                                                          image_width=image_width,
                                                          griding_num=griding_num,
                                                          row_anchor=row_anchor,
                                                          num_lanes=num_lanes,
                                                          use_aux=use_aux)

        self.val_loader_target = datautils.create_loader(val_dataset_target, batch_size, is_train=False, num_workers=num_workers)

        # Test Dataset
        test_dataset = datautils.create_dataset_lbd(data_root,
                                                    target_test,
                                                    image_height=image_height,
                                                    image_width=image_width,
                                                    griding_num=griding_num,
                                                    row_anchor=row_anchor,
                                                    num_lanes=num_lanes,
                                                    use_aux=use_aux)

        self.test_loader = datautils.create_loader(test_dataset, batch_size, is_train=False, num_workers=num_workers)

    def load_model(self):
        """
        Setup and load neural network
        """
        trained_model = self.trained_model
        backbone = self.config.model_params.backbone.split("-")[1]
        griding_num = self.griding_num
        num_lanes = self.num_lanes
        use_aux = self.use_aux
        h_samples = self.config.data_params.h_samples
        cls_dim = (griding_num + 1, len(h_samples), num_lanes)
        temp = self.config.model_params.cls_temp

        assert backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

        torch.backends.cudnn.benchmark = True
        model = Net(pretrained=False, backbone=backbone, cls_dim=cls_dim, use_aux=use_aux).cuda()

        # Load PCS_UFLD model
        if trained_model.endswith(".tar"):
            cls_head = CosineClassifier(cls_dim=cls_dim, temp=temp).cuda()
            trained_model_state_dict = torch.load(trained_model, map_location="cpu")

            model_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict["model_state_dict"].items()}
            model.load_state_dict(model_state_dict, strict=False)

            if "cls_state_dict" in trained_model_state_dict:
                cls_state_dict = trained_model_state_dict["cls_state_dict"]
                cls_head.load_state_dict(cls_state_dict)

            print(f"Model successfully loaded from '{trained_model}', val acc = {trained_model_state_dict['val_acc']}\n")

        # Load UFLD model
        elif trained_model.endswith(".pth"):
            cls_head = Classifier(cls_dim=cls_dim).cuda()
            trained_model_state_dict = torch.load(trained_model, map_location="cpu")["model"]
            trained_model_state_dict = {(key.replace("module.", "") if "module." in key else key): value for key, value in trained_model_state_dict.items()}

            model_state_dict = {key: value for key, value in trained_model_state_dict.items() if "model" in key or "aux" in key or "pool" in key}
            model.load_state_dict(model_state_dict, strict=False)

            cls_state_dict = {key: value for key, value in trained_model_state_dict.items() if "cls" in key}
            cls_head.load_state_dict(cls_state_dict, strict=False)

            print(f"Model successfully loaded from '{trained_model}'\n")
        else:
            raise NotImplementedError("Model not supported or not properly loaded")

        model.eval()
        cls_head.eval()

        return model, cls_head

    @torch.no_grad()
    def validate(self, model, cls_head):
        print(f"Writing result file to {self.out_dir}")
        print(f"Model used: {self.trained_model}")
        self.out_file.write(f"Model used: {self.trained_model}\n")

        # loader = {"source": self.val_loader_source, "target": self.val_loader_target}
        loader = {"target": self.test_loader}

        for domain, loader in loader.items():
            results = {}
            reset_metrics(self.metric_dict)

            if self.use_aux:
                tqdm_batch = tqdm(total=len(loader), desc=f"[Validate model on {domain}]")
                for batch_i, (indices, images, labels, seg_labels) in enumerate(loader):
                    images = images.cuda()
                    labels = labels.cuda()
                    seg_labels = seg_labels.cuda()

                    feat, seg_out = model(images)
                    output = cls_head(feat)

                    results["cls_out"] = torch.argmax(output, dim=1)
                    results["cls_label"] = labels
                    results["seg_out"] = torch.argmax(seg_out, dim=1)
                    results["seg_label"] = seg_labels

                    update_metrics(self.metric_dict, results, use_aux=True)

                    tqdm_batch.update()
                tqdm_batch.close()
            else:
                tqdm_batch = tqdm(total=len(loader), desc=f"[Validate model on {domain}]")
                for batch_i, (indices, images, labels) in enumerate(loader):
                    images = images.cuda()
                    labels = labels.cuda()

                    feat = model(images)
                    output = cls_head(feat)

                    results["cls_out"] = torch.argmax(output, dim=1)
                    results["cls_label"] = labels

                    update_metrics(self.metric_dict, results, use_aux=False)

                    tqdm_batch.update()
                tqdm_batch.close()

            if self.use_aux:
                metrics = [f"{me_name}={100. * me_op.get():.3f}%" for me_name, me_op in zip(self.metric_dict["name"], self.metric_dict["op"])]
            else:
                metrics = [f"{me_name}={100. * me_op.get():.3f}%" for me_name, me_op in zip(self.metric_dict["name"][:3], self.metric_dict["op"][:3])]

            print(f"[Validation result on {domain}] acc={metrics}")
            self.out_file.write(f"[Validation result on {domain}] acc={metrics}\n")
        self.out_file.write("\n")
        self.out_file.close()

    def run(self):
        print("start validating...")
        self.load_dataset()
        model, cls_head = self.load_model()
        self.validate(model, cls_head)
