"""
This script is used to edit labels from already annotated images.
"""

import sys
import pygame
import os
import numpy
import json
from PIL import Image
from label_saver import LabelSaver

WINDOW_WIDTH, WINDOW_HEIGHT = (1680, 820)
IMAGE_WIDTH, IMAGE_HEIGHT = (1280, 720)
FPS = 100

red = (255, 0, 0)
green = (0, 200, 0)
blue = (0, 0, 200)
light_blue = (255, 255, 0)
gray = (150, 150, 150)


class LaneLabeler(object):
    def __init__(self, loading_directory, file_name, new_file_name, num_lanes):
        print(__doc__)
        pygame.init()
        pygame.display.set_caption('CARLANE Labeler')
        pygame.mouse.set_cursor(*pygame.cursors.diamond)

        self.loading_directory = loading_directory
        self.file_name = file_name
        self.new_file_name = new_file_name

        if num_lanes == 4:
            self.colormap = [green, blue, red, light_blue]
        elif num_lanes == 2:
            self.colormap = [green, blue]
        else:
            NotImplementedError("Number of lanes is not supported, choose between 2 or 4 lanes!")

        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.font = pygame.font.SysFont('Arial', 22, bold=True, italic=False)

        self.my_image = None
        self.show_intersection_lanes = False
        self.show_lanes = False
        self.show_cursor = False

        self.screen_offset = 50
        self.counter = 0
        self.file_list = []  # list with file paths (line["raw_data"])
        self.label_file = []  # label list with all data
        self.lane_list = [[] for _ in range(len(self.colormap))]
        self.h_samples = [y for y in range(160, 720, 10)]

        # Lane index from left to right
        self.lane = 0

        self.label_saver = LabelSaver(os.path.join(self.loading_directory, self.new_file_name))

    def render_surface(self, load_file):
        self.window.fill(gray)
        self.my_image = pygame.image.load(os.path.join(self.loading_directory, load_file))
        self.window.blit(self.my_image, (self.screen_offset, self.screen_offset))
        self.window.blit(self.font.render(f'{self.file_list[self.counter]}, Line: {self.counter + 1}', True, (0, 0, 0)), (1280 / 3 + self.screen_offset, 20))
        self.window.blit(self.font.render("Just draw lanes", True, (0, 0, 0)), (IMAGE_WIDTH + self.screen_offset * 2, self.screen_offset))
        self.window.blit(self.font.render("-> automatically sorted", True, (0, 0, 0)), (IMAGE_WIDTH + self.screen_offset * 2, 40 + self.screen_offset))
        self.window.blit(self.font.render("W = Switch lanes", True, (0, 0, 0)), (IMAGE_WIDTH + self.screen_offset * 2, 120 + self.screen_offset))
        self.window.blit(self.font.render("S = Show Cursor", True, (0, 0, 0)), (IMAGE_WIDTH + self.screen_offset * 2, 160 + self.screen_offset))
        self.window.blit(self.font.render("Space = Save data to file", True, (0, 0, 0)), (IMAGE_WIDTH + self.screen_offset * 2, 200 + self.screen_offset))
        self.window.blit(self.font.render("Q = Toggle lines", True, (0, 0, 0)), (IMAGE_WIDTH + self.screen_offset * 2, 240 + self.screen_offset))
        # self.window.blit(self.font.render("E = Toggle lanepoints", True, (0, 0, 0)), (IMAGE_WIDTH + self.screen_offset * 2, 240 + self.screen_offset))
        self.window.blit(self.font.render("Left click = Draw lane", True, (0, 0, 0)), (IMAGE_WIDTH + self.screen_offset * 2, 280 + self.screen_offset))
        self.window.blit(self.font.render("Right click = Undo lanepoint", True, (0, 0, 0)), (IMAGE_WIDTH + self.screen_offset * 2, 320 + self.screen_offset))

        if self.show_intersection_lanes:
            for h in range(160 + self.screen_offset, 720 + self.screen_offset, 10):
                pygame.draw.line(self.window, red, (self.screen_offset, h), (1280 + self.screen_offset, h), 1)
            # for h in range(IMAGE_WIDTH):
            #    pygame.draw.line(self.window, red, (50, h), (IMAGE_WIDTH - 50, h), 1)

        # if self.show_lanes:
        #     self.show_labels()

    def show_labels(self):
        line = self.label_file[self.counter]
        for i in range(len(line['lanes'])):
            lane = line['lanes'][i]
            for j in range(len(line['h_samples'])):
                pygame.draw.circle(self.window, self.colormap[i], (lane[j] + self.screen_offset, line['h_samples'][j] + self.screen_offset), 3, 2)

    def draw_lanes(self):
        for i in range(len(self.lane_list)):
            if len(self.lane_list[i]) == 1:
                pygame.draw.circle(self.window, self.colormap[i], self.lane_list[i][0], 3, 2)
            if len(self.lane_list[i]) > 1:
                for j in range(len(self.lane_list[i]) - 1):
                    pygame.draw.line(self.window, self.colormap[i], self.lane_list[i][j], self.lane_list[i][j + 1], 6)

    def reset_lane(self):
        self.render_surface(self.file_list[self.counter])

        # initialize lanes of first image
        line = self.label_file[self.counter]
        tmp_lanes = []
        for i in range(len(line['lanes'])):
            lane = line['lanes'][i]
            tmp_lane = []
            for j in range(len(line['h_samples']) - 1, 0, -1):
                if lane[j] != -2:
                    point = (lane[j] + self.screen_offset, line['h_samples'][j] + self.screen_offset)
                    tmp_lane.append(point)
            tmp_lanes.append(tmp_lane)

        self.lane_list = tmp_lanes

        self.lane = 0

    def refresh_screen(self):
        self.render_surface(self.file_list[self.counter])
        self.draw_lanes()

    def draw_line(self):
        color = self.colormap[self.lane]
        mouse_position = pygame.mouse.get_pos()
        pygame.draw.circle(self.window, color, mouse_position, 3, 2)
        if len(self.lane_list[self.lane]) > 0:
            pygame.draw.line(self.window, color, self.lane_list[self.lane][-1], mouse_position, 6)

    def calculate_intersections(self, lane_list):
        x_coord = []
        gap = False

        if len(lane_list) > 2:
            last_point = lane_list[0]
            for xy_val in lane_list:
                if last_point == xy_val:
                    continue
                if xy_val[0] == -1:
                    gap = True
                    continue
                if last_point[0] == -1:
                    gap = True
                    last_point = [0.5 * IMAGE_WIDTH, IMAGE_HEIGHT - 1]
                for y_value in reversed(self.h_samples):
                    if gap and (last_point[1] >= y_value > xy_val[1]):
                        x_coord.append(-2)
                    if (last_point == lane_list[0] and last_point[1] < y_value) and last_point[1] < IMAGE_HEIGHT - 0.8 * IMAGE_HEIGHT:
                        x_coord.append(-2)
                    elif (last_point[1] >= y_value > xy_val[1]) or (last_point == lane_list[0] and last_point[1] < y_value) and last_point[1] >= IMAGE_HEIGHT - 0.8 * IMAGE_HEIGHT:
                        if last_point[1] - xy_val[1] == 0:
                            intersection = last_point[1]
                        else:
                            intersection = xy_val[0] + ((y_value - xy_val[1]) * (last_point[0] - xy_val[0])) / (last_point[1] - xy_val[1])
                        if intersection >= IMAGE_WIDTH or intersection < 0:
                            x_coord.append(-2)
                        else:
                            x_coord.append(int(intersection))
                gap = False
                last_point = xy_val

            while len(x_coord) < len(self.h_samples):
                x_coord.append(-2)
            return list(reversed(x_coord))
        else:
            for _ in self.h_samples:
                x_coord.append(-2)
            return x_coord

    def save_data(self):
        x_lanes_list = []
        lanes = []

        # Subtract offset from lanepoints
        for lane in self.lane_list:
            lane_tmp = []
            for point in lane:
                point = (point[0] - self.screen_offset, point[1] - self.screen_offset)
                lane_tmp.append(point)
            lanes.append(lane_tmp)

        for lane in lanes:
            lane = self.calculate_intersections(lane)
            x_lanes_list.append(lane)

        # sort list according to its respective index
        sorting_dict = {}
        for y in range(len(self.h_samples)-1, 0, -1):
            for i, lane in enumerate(x_lanes_list):
                # if lane is invalid (-2)
                if lane[y] < 0:
                    continue
                else:
                    sorting_dict[i] = lane[y]

                # we found 4 lanepoints, which are all valid
                if i == x_lanes_list[-1]:
                    break

            if len(sorting_dict) == len(x_lanes_list):
                break

        sorting_dict = sorted(sorting_dict.items(), key=lambda x: x[1])
        x_lanes_list = [x_lanes_list[i] for i, v in sorting_dict]

        self.label_saver.add_label(x_lanes_list, self.file_list[self.counter])

    def load_json(self):
        try:
            with open(os.path.join(self.loading_directory, self.file_name)) as file:
                self.label_file = [json.loads(line) for line in file.readlines()]

                if os.path.exists(os.path.join(self.loading_directory, self.new_file_name)):
                    tmp_file = open(os.path.join(self.loading_directory, self.new_file_name), 'r')
                    tmp_lines = [line for line in tmp_file.readlines()]
                    self.counter = len(tmp_lines)
                    tmp_file.close()
                else:
                    self.counter = 0

                # initialize lanes of first image
                line = self.label_file[self.counter]
                tmp_lanes = []
                for i in range(len(line['lanes'])):
                    lane = line['lanes'][i]
                    tmp_lane = []
                    for j in range(len(line['h_samples'])-1, 0, -1):
                        if lane[j] != -2:
                            point = (lane[j] + self.screen_offset, line['h_samples'][j] + self.screen_offset)
                            tmp_lane.append(point)
                    tmp_lanes.append(tmp_lane)

                self.lane_list = tmp_lanes

                self.file_list = [line["raw_file"] for line in self.label_file]

        except IndexError:
            print('Continuing with empty .json file')

    def loop(self):
        while True:
            self.refresh_screen()
            self.draw_line()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.label_saver.close_file()
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_ESCAPE:
                        self.label_saver.close_file()
                        pygame.quit()
                        sys.exit()
                    # elif event.key == pygame.K_a:
                    #     self.counter -= 1
                    #     self.reset_lane()
                    elif event.key == pygame.K_s:
                        self.show_cursor = not self.show_cursor
                        pygame.mouse.set_visible(not self.show_cursor)
                    # elif event.key == pygame.K_d:
                    #     self.counter += 1
                    #     self.reset_lane()
                    elif event.key == pygame.K_w:
                        self.lane += 1
                        self.lane %= len(self.colormap)

                        if len(self.lane_list) < 4:
                            self.lane_list.append([])
                    elif event.key == pygame.K_q:
                        self.show_intersection_lanes = not self.show_intersection_lanes
                    # elif event.key == pygame.K_e:
                    #     self.show_lanes = not self.show_lanes
                    elif event.key == pygame.K_SPACE:
                        self.save_data()
                        self.counter += 1
                        self.reset_lane()
                        self.label_saver.close_file()
                        self.label_saver.open_file(os.path.join(self.loading_directory, self.new_file_name))
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        mouse_position = pygame.mouse.get_pos()
                        self.lane_list[self.lane].append(mouse_position)
                        color = self.colormap[self.lane]
                        if len(self.lane_list[self.lane]) == 1:
                            pygame.draw.circle(self.window, color, mouse_position, 3, 2)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 3 and len(self.lane_list[self.lane]) > 0:
                        self.lane_list[self.lane].pop(-1)
            pygame.display.update()

    def execute(self):
        try:
            self.load_json()
            self.loop()
        finally:
            pygame.quit()
            print('Quitting PyGame...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--loading_directory", type=str, default="data_root/", help="root path of images and labels")
    parser.add_argument("--file_name", type=str, default="labels.json", help="Name of the label file")
    parser.add_argument("--new_file_name", type=str, default="new_labels.json", help="Name of the new label file")
    parser.add_argument("--num_lanes", type=int, default=4, help="Choose between 2 or 4 lanes")
    args = parser.parse_args()

    lane_labeler = LaneLabeler(args.loading_directory, args.file_name, args.new_file_name, args.num_lanes)
    lane_labeler.execute()
