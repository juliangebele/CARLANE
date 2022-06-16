import os
import json


class LabelSaver(object):
    def __init__(self, label_file):
        self.file = None
        self.h_samples = [y for y in range(160, 720, 10)]     # loop through row_anchors

        if os.path.exists(label_file):
            print("File already exists! Appending data to this file.")

        self.open_file(label_file)

    def add_label(self, x_lane_list, image_name):
        line = {
            "lanes": x_lane_list,
            "h_samples": self.h_samples,
            "raw_file": image_name
        }
        line = json.dumps(line)
        self.file.write(line + '\n')

    def open_file(self, file):
        self.file = open(file, 'a')

    def close_file(self):
        self.file.close()
