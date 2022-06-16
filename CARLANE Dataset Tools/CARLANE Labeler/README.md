# CARLANE Labeler

Official implementation of the CARLANE Labeler tool used in the paper "CARLANE: A Lane Detection Benchmark for Unsupervised Domain Adaptation from Simulation to multiple Real-World Domains". 

This tool was developed to manually annotate ground truth road lanes on images. We follow [TuSimple's file format](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection) for generating our labels. 

## Installation

Install neccessary dependencies first:

```bash
pip install -r requirements.txt
```

## Getting started
If you want to create new lane labels, place your images in the `image_folder`, then run the main script with the command below.

```bash
python lane_labeler.py
```

If you want to edit existing lane labels (e.g. clean TuSimple's data), reference the TuSimple dataset in `loading_directory` and `label_file` run the following command:

```bash
python lane_labeler_edit_mode.py
```



NOTE: 
- Create at least 3 points for saving valid lanes!
- The amount of lanes is defined by the colors, stored in the colormap variable (better use 2 or 4 lanes).

All possible commands are displayed on the main window. 