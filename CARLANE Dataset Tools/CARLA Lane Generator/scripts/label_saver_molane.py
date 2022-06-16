import os
import json
import config_molane as cfg


class LabelSaver:
    """
    Helper class to save all the lanedata (labels). Each label contains a list 
    of the x values of a lane, their corresponding predefined y-values and 
    their path to the image.
    """
    def __init__(self, label_file):
        self.buffer = []
        self.image_name = 0
        self.curve_counters = {
            'steep_left_curve/': 0,
            'left_curve/': 0,
            'straight/': 0,
            'right_curve/': 0,
            'steep_right_curve/': 0
        }
        
        folder = os.path.dirname(label_file)
        if not os.path.isdir(folder):
            os.makedirs(folder)
            
        if os.path.exists(label_file):
            print("Label file already exists, appending data to file.")

        self.file = open(label_file, 'a')

    def is_full(self):
        return len(self.buffer) == cfg.number_of_images

    def reset(self):
        self.buffer = []

    def save(self):
        try:
            for line in self.buffer:
                self.file.write(line)
        except IndexError:
            print('WARNING: no full labelbuffer saved')

    def add_label(self, x_lane_list, curve_type):
        if self.is_full():
            self.save()
            self.reset()

        self.image_name = self.curve_counters[curve_type]

        line = {
            'lanes': x_lane_list,
            'h_samples': cfg.h_samples,
            'raw_file': cfg.suffix + curve_type + f'{self.image_name:04d}' + '.jpg'   # train/sim/TownXX/ + right_curve/ + 0001.jpg
        }

        line = json.dumps(line)
        self.buffer.append(line + '\n')

        self.curve_counters[curve_type] += 1

    def close_file(self):
        self.file.close()
