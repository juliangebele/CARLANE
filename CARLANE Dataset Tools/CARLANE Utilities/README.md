# CARLANE Utilities

Official implementation of the CARLANE Utilities used in the paper "CARLANE: A Lane Detection Benchmark for Unsupervised Domain Adaptation from Simulation to multiple Real-World Domains". 

This repository contains several tools to support the creation of the sim-to-real datasets. 

## Overview

Some scripts support command line arguments. 

### 1. compress_jpg.py
Compress .jpg images.

### 2. create_datalist.py
Creates the .txt file, used for unlabeled images, i.e. train images from the target domain.

### 3. create_segmentation_labels_2lanes.py/ create_segmentation_labels_4lanes.py
Create the .png labels with .txt files used for the labeled train, validation and test data. Both scripts differ in the number of lanes. (MoLane: 2 lanes, TuLane & MuLane: 4 lanes)
- root: root directory where the images are placed
- label_files: list of label files (paths), pointing to the raw .json label file
- new_file: whole path and name of the new .txt file, which is generated from the raw labels

### 4. create_tusimple_reduced_dataset.py
Keep all necessary image (20.jpg) files and remove all other images.

### 5. create_video.py
Create a video from images in a folder. 

### 6. dataset_balancer_3bins.py/ dataset_balancer_5bins.py
Balances the images across its curve classes, that every class contains the same amount of images, only used for Town04 and Town06. TuLane: 3 bins (left, straight, right curve), MoLane: 5 bins (hard left, soft left, straight, soft right and hard right curve). Command line arguments:
```
--town_name: name of the Town folder, whose images are balanced
--root_dir: root directory where the images are placed
--label_root_path: root directory where the labels are placed
--imgs_per_dir: amount of images per folder
```
You can override parameters like
```
python dataset_balancer_3bins.py --town_name Town06 --root_dir ./MoLane/data/train/sim/
```

### 7. multi_DA_MoLane_real_train_sampler.py
Samples the image data from MoLane's real train images to construct MuLane's first target domain. 
Command line arguments:
```
--root_dir: root directory where the images are placed
--root: root directory of the source dataset
--target_dir_root: root directory of the target dataset
--real_train_file_name: name of the new label file
--n_samples: number of images per folder to sample
```

### 8. multi_DA_simulation_data_sampler.py
Samples the image data from MoLane's source domain to construct MuLane's first source domain. 
Command line arguments:
```
--sim_labels: path to simulation labels
--root_dir: root directory where the images are placed
--root: root directory of the source dataset
--target_dir_root: root directory of the target dataset
--new_label_file_name: name of the new label file
```

### 9. multi_DA_TuSimple_val_test_sampler.py
Samples the data from TuSimple's validation and test data to construct MuLane's validation and test data. 
Command line arguments:
```
--real_val_file_name: TuSimple's validation labels
--real_test_file_name: TuSimple's test labels
--new_real_val_file_name: new validation file sampled from TuSimple
--new_real_test_file_name: new test file sampled from TuSimple
```

### 10. rename_files.py
Rename image files or reverse order of images in a folder. 
