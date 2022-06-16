#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
==============================================
     Welcome to CARLA Lane Detection.

     Manuals Controls so far:

         Press Space to toggle Saving
         Press R to reset vehicle position
         Press Esc to quit
==============================================
"""

# ==============================================================================
#  Table of Contents 
#   1. Imports
#   2. Global Variables
#   3. Global Methods
#   4. Classes
#       4.1 CarlaSyncMode
#       4.2 CarlaGame
#       4.3 VehicleManager
#       4.4 Lanemarkings
#   5. Main
# ==============================================================================

# ==============================================================================
# -- 1. Imports ----------------------------------------------------------------
# ==============================================================================

import glob
import os
import sys
import math
import random
import pygame
import numpy as np
import itertools
from collections import deque
from scripts.image_saver import ImageSaver
from scripts.label_saver_tulane import LabelSaver
import config_tulane as cfg

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

try:
    import queue
except ImportError:
    import Queue as queue

print(__doc__)

# ==============================================================================
# -- 2. Global Variables -------------------------------------------------------
# ==============================================================================

IMAGE_WIDTH = cfg.IMAGE_WIDTH
IMAGE_HEIGHT = cfg.IMAGE_HEIGHT
FPS = cfg.FPS
FOV = cfg.FOV

len_of_lanepoints = int(cfg.number_of_lanepoints*(1/cfg.meters_per_frame))

if cfg.is_solo_lane:
    lanes = [deque(maxlen=len_of_lanepoints),
             deque(maxlen=len_of_lanepoints)]
else:
    lanes = [deque(maxlen=len_of_lanepoints),
             deque(maxlen=len_of_lanepoints),
             deque(maxlen=len_of_lanepoints),
             deque(maxlen=len_of_lanepoints)]


# ==============================================================================
# -- 3. Global Methods ---------------------------------------------------------
# ==============================================================================

def reshape_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    return array


def draw_image(surface, image, blend=False):
    array = reshape_image(image)
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


# ==============================================================================
# -- 4.1 CarlaSyncMode ---------------------------------------------------------
# ==============================================================================

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context.

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)
    """
    def __init__(self, world, *sensors, **kwargs):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / kwargs.get('fps', FPS)
        self._queues = []
        self._settings = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(no_rendering_mode=False, synchronous_mode=True, fixed_delta_seconds=self.delta_seconds))

        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


# ==============================================================================
# -- 4.2 Carla Game ------------------------------------------------------------
# ==============================================================================

class CarlaGame(object):
    """
    Main Game Instance to execute carla simulator in pygame.
    """
    def __init__(self):
        self.actor_list = []
        pygame.init()

        self.display = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(5.0)

        # self.world = self.client.get_world()
        self.world = self.client.load_world(cfg.CARLA_TOWN)
        self.map = self.world.get_map()

        # Hide all objects on the map and show street only
        if cfg.CARLA_TOWN.endswith('_Opt'):
            map_layer_names = [
                carla.MapLayer.NONE,
                carla.MapLayer.Buildings,
                carla.MapLayer.Decals,
                carla.MapLayer.Foliage,
                carla.MapLayer.Ground,
                carla.MapLayer.ParkedVehicles,
                carla.MapLayer.Particles,
                carla.MapLayer.Props,
                carla.MapLayer.StreetLights,
                carla.MapLayer.Walls,
                carla.MapLayer.All
            ]

            for layer in map_layer_names:
                self.world.unload_map_layer(layer)
            self.world.load_map_layer(carla.MapLayer.StreetLights)

        self.lane_markings = LaneMarkings()
        self.vehicle_manager = VehicleManager()
        self.image_saver = ImageSaver(cfg.saving_directory, cfg.number_of_images, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
        self.label_saver = LabelSaver(f'{cfg.root}{cfg.CARLA_TOWN}.json')

        self.sync_mode = None
        self.vehicle = None
        self.camera_transform = None
        self.camera_rgb = None
        self.camera_semseg = None
        self.waypoint_list = None
        self.waypoints_visited = None
        self.is_saving = cfg.is_saving
        self.image_counter = 0

        self.curve_counters = {
            'left_curve/': 0,
            'straight/': 0,
            'right_curve/': 0
        }

        self.filter_dict = {
            'Town03': {
                'invalid_road_ids': [18, 48, 49, 50, 51, 52, 60, 76, 77, 78, 79, 80],
                'waypoint_lengths': 45
            },
            'Town03_Opt': {
                'invalid_road_ids': [18, 48, 49, 50, 51, 52, 60, 76, 77, 78, 79, 80],
                'waypoint_lengths': 45
            },
            'Town04': {
                'invalid_road_ids': [38, 51, 52, 1602],
                'waypoint_lengths': 45
            },
            'Town04_Opt': {
                'invalid_road_ids': [38, 51, 52, 1602],
                'waypoint_lengths': 45
            },
            'Town05': {
                'invalid_road_ids': [18, 19, 20, 40, 368],
                'waypoint_lengths': 45
            },
            'Town05_Opt': {
                'invalid_road_ids': [18, 19, 20, 40, 368],
                'waypoint_lengths': 45
            },
            'Town06': {
                'invalid_road_ids': [7, 21, 43, 400, 834, 1139],
                'waypoint_lengths': 40
            },
            'Town06_Opt': {
                'invalid_road_ids': [7, 21, 43, 400, 834, 1139],
                'waypoint_lengths': 40
            },
            'Town10HD': {
                'invalid_road_ids': [515],
                'waypoint_lengths': 40
            },
            'Town10HD_Opt': {
                'invalid_road_ids': [515],
                'waypoint_lengths': 40
            }
        }

        if cfg.is_third_person:
            self.camera_transforms = [carla.Transform(carla.Location(x=-4.5, z=2.2), carla.Rotation(pitch=-14.5)),
                                      carla.Transform(carla.Location(x=-4.0, z=2.2), carla.Rotation(pitch=-18.0))]
        else:
            self.camera_transforms = [carla.Transform(carla.Location(x=1.4, y=-0.2, z=1.7), carla.Rotation(pitch=-11.5)),
                                      carla.Transform(carla.Location(x=1.4, y=-0.38, z=1.63), carla.Rotation(pitch=-11.5)),
                                      carla.Transform(carla.Location(x=1.4, y=-0.35, z=1.60), carla.Rotation(pitch=-11.2))]

    def reset_vehicle_position(self):
        """
        Resets the vehicles' position on the map. After reset the agent creates 
        a new route of (len_of_lanepoints) waypoints to follow along.
        """
        start_position = random.choice(self.map.get_spawn_points())
        waypoint = self.map.get_waypoint(start_position.location)

        # Initialize lane deques with a fixed number of lanepoints
        for lane in lanes:
            for lanepoint in range(cfg.number_of_lanepoints):
                lane.append(None)

        # Create n waypoints to have an initial route for the vehicle
        self.waypoint_list = deque(maxlen=len_of_lanepoints)
        self.waypoints_visited = deque(maxlen=cfg.number_of_points_visited)

        for i in np.linspace(cfg.meters_per_frame, cfg.number_of_lanepoints, len_of_lanepoints):
            self.waypoint_list.append(waypoint.next(i)[0])

        # Last 10 points already visited
        self.waypoints_visited.append(waypoint)

        for lanepoint in self.waypoint_list:
            lane_markings = self.lane_markings.calculate_3d_lanepoints(lanepoint)

        camera_position = random.choice(self.camera_transforms)
        self.camera_rgb.set_transform(camera_position)
        self.camera_semseg.set_transform(camera_position)

        # Random rotation angles for agent
        self.vehicle_manager.angle = random.choice(cfg.rotation_angles)

        # Draw all 3D lanemarkings in carla simulator
        if cfg.draw_lanes:
            for i, color in enumerate(self.lane_markings.colormap_carla):
                for j in range(cfg.number_of_lanepoints - 1):
                    self.lane_markings.draw_lanes(self.client, lane_markings[i][j], lane_markings[i][j + 1], self.lane_markings.colormap_carla[color])

    def detect_lanemarkings(self, new_waypoint, image_semseg):
        """
        Calculate and show all lanes on the road.

        Args:
            new_waypoint: List of carla.Waypoints. Calculate all the lanemarkings based on the new_waypoint, which is the last element from the waypoint_list.
            image_semseg: numpy array. Filter lanepoints with semantic segmentation camera. 
        """
        lanes_list = []  # filtered 2D-Points
        x_lanes_list = []  # only x values of lanes
        for lane_coords in self.lane_markings.calculate_3d_lanepoints(new_waypoint):
            lane = self.lane_markings.calculate_2d_lanepoints(self.camera_rgb, lane_coords)
            lane = self.lane_markings.calculate_intersections(lane)
            lane = self.lane_markings.filter_2d_lanepoints(lane, image_semseg)
            lanes_list.append(lane)
            x_lanes_list.append(self.lane_markings.format_2d_lanepoints(lane))

        return lanes_list, x_lanes_list

    def render_display(self, image, image_semseg, lanes_list, render_lanes=True):
        """
        Renders the images captured from both cameras and shows it on the
        pygame display

        Args:
            image: numpy array. Shows the 3-channel numpy imagearray on the pygame display.
            image_semseg: numpy array. Shows the semantic segmentation image on the pygame display.
            lanes_list: list of the lanes that are generated
            render_lanes: show lanes as points on pygame display
        """
        # Draw the display.
        draw_image(self.display, image)
        # draw_image(self.display, image_semseg, blend=True)

        # Draw lanepoints of every lane on pygame window
        if render_lanes:
            for i, color in enumerate(self.lane_markings.colormap):
                if lanes_list[i]:
                    for j in range(len(lanes_list[i])):
                        pygame.draw.circle(self.display, self.lane_markings.colormap[color], lanes_list[i][j], 3, 2)

        self.display.blit(self.font.render('% 5d FPS ' % self.clock.get_fps(), True, (255, 255, 255)), (8, 10))
        self.display.blit(self.font.render('Map: ' + cfg.CARLA_TOWN, True, (255, 255, 255)), (20, 30))
        self.display.blit(self.font.render('Curves: ' + f'{list(self.curve_counters.values())}', True, (255, 255, 255)), (20, 50))
        self.display.blit(self.font.render('Image: ' + f'{self.image_counter}', True, (255, 255, 255)), (20, 70))
        self.display.blit(self.font.render('Road-ID: ' + f'{self.waypoint_list[0].road_id}', True, (255, 255, 255)), (20, 90))

        pygame.display.flip()

    def execute(self):
        try:
            self.initialize()
            with CarlaSyncMode(self.world, self.camera_rgb, self.camera_semseg, fps=FPS) as self.sync_mode:
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            sys.exit()
                        elif event.type == pygame.KEYUP:
                            if event.key == pygame.K_ESCAPE:
                                pygame.quit()
                                sys.exit()
                            elif event.key == pygame.K_r:
                                self.reset_vehicle_position()
                            elif event.key == pygame.K_SPACE:
                                self.is_saving = not self.is_saving
                    if self.stop_saving():
                        return
                    self.on_gameloop()
        finally:
            self.image_saver.save()
            self.label_saver.save()
            self.label_saver.close_file()

            print('Image data collected')
            print('Destroying actors and cleaning up.')
            for actor in self.actor_list:
                actor.destroy()

            self.vehicle_manager.destroy()

            pygame.quit()
            print('Quitting')

    def initialize(self):
        """
        Initialize the world and spawn vehicles, cameras and saver.
        """
        blueprint_library = self.world.get_blueprint_library()

        start_position = random.choice(self.map.get_spawn_points())
        self.vehicle = self.world.spawn_actor(random.choice(blueprint_library.filter('vehicle.bh.crossbike')), start_position)
        self.actor_list.append(self.vehicle)
        self.vehicle.set_simulate_physics(False)
        self.camera_transform = self.camera_transforms[random.randint(0, len(self.camera_transforms) - 1)]

        # Spawn rgb-cam and attach to vehicle
        bp_camera_rgb = blueprint_library.find('sensor.camera.rgb')
        bp_camera_rgb.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
        bp_camera_rgb.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
        bp_camera_rgb.set_attribute('fov', f'{FOV}')
        camera_rgb_spawnpoint = self.camera_transform
        self.camera_rgb = self.world.spawn_actor(bp_camera_rgb, camera_rgb_spawnpoint, attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        # Spawn semseg-cam and attach to vehicle
        bp_camera_semseg = blueprint_library.find('sensor.camera.semantic_segmentation')
        bp_camera_semseg.set_attribute('image_size_x', f'{IMAGE_WIDTH}')
        bp_camera_semseg.set_attribute('image_size_y', f'{IMAGE_HEIGHT}')
        bp_camera_semseg.set_attribute('fov', f'{FOV}')
        camera_semseg_spawnpoint = self.camera_transform
        self.camera_semseg = self.world.spawn_actor(bp_camera_semseg, camera_semseg_spawnpoint, attach_to=self.vehicle)
        self.actor_list.append(self.camera_semseg)

        self.reset_vehicle_position()

        # Spawn five random vehicles around the car to create a realistic traffic scenario
        self.vehicle_manager.spawn_vehicles(self.world)

    def on_gameloop(self):
        """
        Determines the logic of movement and what should happen every frame. 
        Also adds an image to the image saver and the corresponding label to the label saver, if the frame is meant to be saved. 
        In the actual implementation the points, which will be saved, are drawn to the screen on runtime (not on the saved images).
        """
        self.clock.tick()

        # Advance the simulation and wait for the data.
        snapshot, image_rgb, image_semseg = self.sync_mode.tick(timeout=1.0)

        try:
            # Move own vehicle to the next waypoint
            new_waypoint, curve_type = self.vehicle_manager.move_agent(self.vehicle, self.waypoint_list, self.waypoints_visited)

            # Move neighbor vehicles with the same speed as the own vehicle
            self.vehicle_manager.move_vehicles(self.waypoint_list)

            # Convert and reshape image from (720*1280*3)x1 to shape(720, 1280, 3)
            image_semseg.convert(carla.ColorConverter.CityScapesPalette)
            image_semseg = reshape_image(image_semseg)

            # Calculate all the lanes
            lanes_list, x_lanes_list = self.detect_lanemarkings(new_waypoint, image_semseg)

            # Draw all 3D lanes in carla simulator
            if cfg.draw_lanes:
                for i, color in enumerate(self.lane_markings.colormap_carla):
                    self.lane_markings.draw_lanes(self.client, lanes[i][-1], lanes[i][-2], self.lane_markings.colormap_carla[color])

            # Render the pygame display and show the lanes accordingly
            self.render_display(image_rgb, image_semseg, lanes_list)

            # Save images using imagesaver
            if self.is_saving and self.save_valid_lanes() and self.filter_lanes(x_lanes_list) and self.filter_junctions():
                if curve_type != 'straight/' or (curve_type == 'straight/' and self.curve_counters['straight/'] < max(self.curve_counters.values())):
                    self.save_data(image_rgb, x_lanes_list, curve_type)
                if self.image_counter % cfg.reset_intervall == 0 and self.curve_counters[curve_type] < cfg.max_images_per_class:
                    self.reset_vehicle_position()

        except IndexError:
            print("No waypoints to choose from, resetting")
            self.reset_vehicle_position()

    def save_data(self, image_rgb, x_lanes_list, curve_type):
        if cfg.CARLA_TOWN == 'Town06' or cfg.CARLA_TOWN == 'Town06_Opt':
            if (curve_type == 'left_curve/' or curve_type == 'right_curve/') and \
                    self.curve_counters['left_curve/'] + self.curve_counters['right_curve/'] < cfg.max_images_per_class * 2:
                self.image_saver.add_image(image_rgb.raw_data, 'CameraRGB', curve_type)
                self.label_saver.add_label(x_lanes_list, curve_type)
                self.curve_counters[curve_type] += 1
                self.image_counter += 1
            elif curve_type == 'straight/' and self.curve_counters['straight/'] < cfg.max_images_per_class:
                self.image_saver.add_image(image_rgb.raw_data, 'CameraRGB', curve_type)
                self.label_saver.add_label(x_lanes_list, curve_type)
                self.curve_counters[curve_type] += 1
                self.image_counter += 1
        else:
            if self.curve_counters[curve_type] < cfg.max_images_per_class:
                self.image_saver.add_image(image_rgb.raw_data, 'CameraRGB', curve_type)
                self.label_saver.add_label(x_lanes_list, curve_type)
                self.curve_counters[curve_type] += 1
                self.image_counter += 1

    def stop_saving(self):
        return True if self.image_counter >= cfg.max_images_per_class * len(self.curve_counters) else False

    def save_valid_lanes(self):
        # Only save images with clean labels
        return False if self.waypoint_list[0].road_id in self.filter_dict[cfg.CARLA_TOWN]['invalid_road_ids'] else True

    def filter_junctions(self):
        waypoints = [i for i in itertools.islice(self.waypoint_list, self.filter_dict[cfg.CARLA_TOWN]['waypoint_lengths'])]
        for waypoint in waypoints:
            if waypoint.is_junction:
                return False
        return True

    @staticmethod
    def filter_lanes(x_lanes_list):
        # Filter images with no labels
        is_lane = [False] * len(x_lanes_list)
        for i in range(len(x_lanes_list)):
            if max(x_lanes_list[i]) > -2:
                is_lane[i] = True
        return max(is_lane)


# ==============================================================================
# -- 4.3 Vehicle Manager -------------------------------------------------------
# ==============================================================================

class VehicleManager(object):
    """
    Helper class to spawn and manage neighbor vehicles and
    detect junctions and curves
    """
    def __init__(self):
        self.deviation_counter = 0.0
        self.angle = 0
        self.oscillation = 0.0
        self.curve_type = ''
        self.vehicles_list = []
        self.vehicleswap_counter = 0
        self.indices_of_vehicles = []
        self.waypoint_indices = []
        self.transforms = None

        # for Town03, Town05, Town10 due to junctions use 2
        self.actual_pos = 0

    def detect_curve(self, vehicle, waypoint_list):
        """
        Detect the direction of the lane. Using a point at the start and in the middle to check if there is
        a curve. Cross product is used to check if left or right. Dot product is used to check when we have
        a curve.
        """
        lane_angle = cfg.lane_angle
        waypoint_distance = cfg.waypoint_distance
        vector_start = [vehicle.get_transform().get_forward_vector().x, vehicle.get_transform().get_forward_vector().y]
        vector_end = [waypoint_list[waypoint_distance].transform.get_forward_vector().x, waypoint_list[waypoint_distance].transform.get_forward_vector().y]

        unit_vector_start = vector_start / np.linalg.norm(vector_start)
        unit_vector_end = vector_end / np.linalg.norm(vector_end)
        cross_product = np.cross(unit_vector_start, unit_vector_end)
        dot_product = np.dot(unit_vector_start, unit_vector_end)
        angle = np.arccos(np.clip(dot_product, -1, 1))
        angle = math.degrees(angle)

        if angle < lane_angle['straight']:
            self.curve_type = 'straight/'
            return False
        elif angle >= lane_angle['curve']:
            return False
        else:
            if cross_product < 0:
                self.curve_type = 'left_curve/'
                return True
            elif cross_product > 0:
                self.curve_type = 'right_curve/'
                return True
            else:
                self.curve_type = 'straight/'
                return False

    def move_agent(self, vehicle, waypoint_list, waypoints_visited):
        """
        Move the own agent along the given waypoints in waypoint_list.
        After we moved to the first position of the list, we need to look 
        out for a new future waypoint to append to the list, which is chosen 
        randomly. The waypoint_list always contains a fixed number of waypoints.

        Args:
            vehicle: carla.Actor. Vehicle to move along the given waypoints.
            waypoint_list: list. List of waypoints to move the vehicle along.
            waypoints_visited: get the rotation of the waypoint 10m behind to get more diverse labels

        Returns:
            new_waypoint: carla.Waypoint. Append the new_waypoint to the waypoint_list every tick.
        """
        self.deviation_counter += 0.08

        if self.detect_curve(vehicle, waypoint_list):
            self.oscillation = 0.5
        else:
            self.oscillation = 0.5

        vehicle.set_transform(carla.Transform(waypoint_list[self.actual_pos].transform.location +
                                              waypoint_list[self.actual_pos].transform.get_right_vector() *
                                              self.oscillation * (2 / math.pi * math.asin(math.sin(self.deviation_counter))),
                                              carla.Rotation(pitch=waypoint_list[0].transform.rotation.pitch,
                                                             yaw=waypoints_visited[0].transform.rotation.yaw + self.angle * math.sin(self.deviation_counter),
                                                             roll=waypoint_list[0].transform.rotation.roll)))

        # Finally look for a list of future waypoints to append to the list and show the lanepoints accordingly
        potential_new_waypoints = waypoint_list[-1].next(cfg.meters_per_frame)
        new_waypoint = random.choice(potential_new_waypoints)
        waypoint_list.append(new_waypoint)

        # fill the list every tick with our actual position
        waypoints_visited.append(waypoint_list[0])

        return new_waypoint, self.curve_type

    def spawn_vehicles(self, world):
        """
        Spawns 5 random vehicles on the map. 
        Speed of the vehicles will be adjusted later, when they 
        are "attached" to the own car.

        Args:
            world: carla.World. Get the spawnpoints from the world object.
        """

        self.transforms = [carla.Transform(carla.Location(-1000, -1000, 0)),
                           carla.Transform(carla.Location(-1000, -1010, 0)),
                           carla.Transform(carla.Location(-1000, -1020, 0)),
                           carla.Transform(carla.Location(-1000, -1030, 0)),
                           carla.Transform(carla.Location(-1000, -1040, 0))]

        spawn_points = world.get_map().get_spawn_points()
        vehicles = world.get_blueprint_library().filter('vehicle.*')
        cars = [vehicle for vehicle in vehicles if int(vehicle.get_attribute('number_of_wheels')) == 4]
        random.shuffle(cars)

        for i, car in enumerate(cars[:5]):
            neighbor_vehicle = world.spawn_actor(car, spawn_points[i])
            neighbor_vehicle.set_simulate_physics(False)
            neighbor_vehicle.set_transform(self.transforms[i])
            self.vehicles_list.append(neighbor_vehicle)

    def move_vehicles(self, waypoint_list, frame_counter=50):
        """
        Move the neighbor vehicles with the same speed as the own vehicle. 
        Methods are encapsulated in nested function to prevent calling
        them from outside, since they only should be called all together 
        and not individually.

        Args:
            waypoint_list: list. Access the waypoints from the waypoint_list to check lane information.
            frame_counter: int. Use the frame_counter to determine, when to randomly swap the neighbor vehicles. 
        """

        def move_mid_vehicle():
            if waypoint_list[self.waypoint_indices[0]]:
                next_transform = carla.Transform(waypoint_list[self.waypoint_indices[0]].transform.location,
                                                 waypoint_list[self.waypoint_indices[0]].transform.rotation)
                self.vehicles_list[0].set_transform(next_transform)
            else:
                self.vehicles_list[0].set_transform(self.transforms[0])

        def move_far_left_vehicle():
            if (waypoint_list[self.waypoint_indices[1]].get_left_lane() and
                    waypoint_list[self.waypoint_indices[1]].get_left_lane().lane_type == carla.LaneType.Driving and
                    waypoint_list[self.waypoint_indices[1]].get_left_lane().transform.rotation == waypoint_list[self.waypoint_indices[1]].transform.rotation):
                next_transform = carla.Transform(waypoint_list[self.waypoint_indices[1]].get_left_lane().transform.location,
                                                 waypoint_list[self.waypoint_indices[1]].get_left_lane().transform.rotation)
                self.vehicles_list[1].set_transform(next_transform)
            else:
                self.vehicles_list[1].set_transform(self.transforms[1])

        def move_far_right_vehicle():
            if (waypoint_list[self.waypoint_indices[2]].get_right_lane() and
                    waypoint_list[self.waypoint_indices[2]].get_right_lane().lane_type == carla.LaneType.Driving and
                    waypoint_list[self.waypoint_indices[2]].get_right_lane().transform.rotation == waypoint_list[self.waypoint_indices[2]].transform.rotation):
                next_transform = carla.Transform(waypoint_list[self.waypoint_indices[2]].get_right_lane().transform.location,
                                                 waypoint_list[self.waypoint_indices[2]].get_right_lane().transform.rotation)
                self.vehicles_list[2].set_transform(next_transform)
            else:
                self.vehicles_list[2].set_transform(self.transforms[2])

        def move_close_left_vehicle():
            if (waypoint_list[self.waypoint_indices[3]].get_left_lane() and
                    waypoint_list[self.waypoint_indices[3]].get_left_lane().lane_type == carla.LaneType.Driving and
                    waypoint_list[self.waypoint_indices[3]].get_left_lane().transform.rotation == waypoint_list[self.waypoint_indices[3]].transform.rotation):
                next_transform = carla.Transform(waypoint_list[self.waypoint_indices[3]].get_left_lane().transform.location,
                                                 waypoint_list[self.waypoint_indices[3]].get_left_lane().transform.rotation)
                self.vehicles_list[3].set_transform(next_transform)
            else:
                self.vehicles_list[3].set_transform(self.transforms[3])

        def move_close_right_vehicle():
            if (waypoint_list[self.waypoint_indices[4]].get_right_lane() and
                    waypoint_list[self.waypoint_indices[4]].get_right_lane().lane_type == carla.LaneType.Driving and
                    waypoint_list[self.waypoint_indices[4]].get_right_lane().transform.rotation == waypoint_list[self.waypoint_indices[4]].transform.rotation):
                next_transform = carla.Transform(waypoint_list[self.waypoint_indices[4]].get_right_lane().transform.location,
                                                 waypoint_list[self.waypoint_indices[4]].get_right_lane().transform.rotation)
                self.vehicles_list[4].set_transform(next_transform)
            else:
                self.vehicles_list[4].set_transform(self.transforms[4])

        # Each move function is being mapped with an index, so we just use the index to move a neighbor vehicle
        movement_map = {
            0: move_mid_vehicle,
            1: move_far_left_vehicle,
            2: move_far_right_vehicle,
            3: move_close_left_vehicle,
            4: move_close_right_vehicle
        }

        for i in self.indices_of_vehicles:
            movement_map[i]()

        # Randomly swap the cars
        if self.vehicleswap_counter % frame_counter == 0:
            # How many cars to spawn?
            number_of_vehicles = random.randint(0, 5)
            # Which vehicles' index?
            self.indices_of_vehicles = random.sample(range(5), number_of_vehicles)
            # Where to spawn at?
            self.waypoint_indices = [random.randint(int((1/6)*len_of_lanepoints), int((1/2)*len_of_lanepoints)),
                                     random.randint(int((1/3)*len_of_lanepoints), int((1/2)*len_of_lanepoints)),
                                     random.randint(int((1/3)*len_of_lanepoints), int((1/2)*len_of_lanepoints)),
                                     random.randint(0, int((1/5)*len_of_lanepoints)),
                                     random.randint(0, int((1/5)*len_of_lanepoints))]

            # Despawn them for a short moment to "clean up" road
            for i, vehicle in enumerate(self.vehicles_list):
                vehicle.set_transform(self.transforms[i])

        self.vehicleswap_counter += 1

    def destroy(self):
        for vehicle in self.vehicles_list:
            vehicle.destroy()


# ==============================================================================
# -- 4.4 Lane Markings ---------------------------------------------------------
# ==============================================================================

class LaneMarkings(object):
    """
    Helper class to detect and draw lanemarkings in carla.
    """
    def __init__(self):
        if cfg.is_solo_lane:
            self.colormap = {
                'green': (0, 255, 0),
                'red': (255, 0, 0)
            }

            self.colormap_carla = {
                'green': carla.Color(0, 255, 0),
                'red': carla.Color(255, 0, 0)
            }
        else:
            self.colormap = {
                'green': (0, 255, 0),
                'red': (255, 0, 0),
                'yellow': (255, 255, 0),
                'blue': (0, 0, 255)
            }

            self.colormap_carla = {
                'green': carla.Color(0, 255, 0),
                'red': carla.Color(255, 0, 0),
                'yellow': carla.Color(255, 255, 0),
                'blue': carla.Color(0, 0, 255)
            }

        # Intrinsic camera matrix needed to convert 3D-world coordinates to 2D-imagepoints
        f = IMAGE_WIDTH / (2 * math.tan(FOV * math.pi / 360))
        c_x = IMAGE_WIDTH / 2
        c_y = IMAGE_HEIGHT / 2

        self.camera_matrix = np.float32([[f, 0, c_x],
                                         [0, f, c_y],
                                         [0, 0, 1]])

    @staticmethod
    def draw_points(client, point):
        client.get_world().debug.draw_point(point + carla.Location(z=0.05), size=0.05, life_time=cfg.number_of_lanepoints / FPS, persistent_lines=False)

    @staticmethod
    def draw_lanes(client, point0, point1, color):
        if point0 and point1:
            client.get_world().debug.draw_line(point0 + carla.Location(z=0.05), point1 + carla.Location(z=0.05), thickness=0.05,
                                               color=color, life_time=cfg.number_of_lanepoints / FPS, persistent_lines=False)

    @staticmethod
    def calculate_3d_lanepoints(lanepoint):
        """
        Calculates the 3D position of the lane.
        Information being used is: lanemarking type, lane type, lane width and driving direction of the lanepoint.
        
        Args:
            lanepoint: carla.Waypoint. The lanepoint, which the lanemarkings should be calculated of.

        Returns:
            lanes: list of lanes. 3D-positions of every lanemarking of the given lanepoints actual lane, and corresponding neighbour lanes if they exist.
        """
        rotation_vec = lanepoint.transform.get_forward_vector()

        length = math.sqrt(rotation_vec.y ** 2 + rotation_vec.x ** 2)
        length_vec = carla.Location(rotation_vec.y, -rotation_vec.x, 0) / length * 0.5 * lanepoint.lane_width
        right_lanemarking = lanepoint.transform.location - length_vec
        left_lanemarking = lanepoint.transform.location + length_vec

        if cfg.junction_mode:
            lanes[0].append(left_lanemarking) if (lanepoint.left_lane_marking.type != carla.LaneMarkingType.NONE) else lanes[0].append(None)
            lanes[1].append(right_lanemarking) if (lanepoint.right_lane_marking.type != carla.LaneMarkingType.NONE) else lanes[1].append(None)

            if not cfg.is_solo_lane:
                # Calculate remaining outer lanes (left and right).
                if lanepoint.get_left_lane() and lanepoint.get_left_lane().left_lane_marking.type != carla.LaneMarkingType.NONE:
                    outer_left_lanemarking = lanepoint.transform.location + 3 * length_vec
                    lanes[2].append(outer_left_lanemarking)
                else:
                    lanes[2].append(None)

                if lanepoint.get_right_lane() and lanepoint.get_right_lane().right_lane_marking.type != carla.LaneMarkingType.NONE:
                    outer_right_lanemarking = lanepoint.transform.location - 3 * length_vec
                    lanes[3].append(outer_right_lanemarking)
                else:
                    lanes[3].append(None)
        else:
            lanes[0].append(left_lanemarking) if left_lanemarking else lanes[0].append(None)
            lanes[1].append(right_lanemarking) if right_lanemarking else lanes[1].append(None)

            if not cfg.is_solo_lane:
                # Calculate remaining outer lanes (left and right).
                if lanepoint.get_left_lane() and lanepoint.get_left_lane().lane_type == carla.LaneType.Driving:
                    outer_left_lanemarking = lanepoint.transform.location + 3 * length_vec
                    lanes[2].append(outer_left_lanemarking)
                else:
                    lanes[2].append(None)

                if lanepoint.get_right_lane() and lanepoint.get_right_lane().lane_type == carla.LaneType.Driving:
                    outer_right_lanemarking = lanepoint.transform.location - 3 * length_vec
                    lanes[3].append(outer_right_lanemarking)
                else:
                    lanes[3].append(None)

        return lanes

    def calculate_2d_lanepoints(self, camera_rgb, lane_list):
        """
        Transforms the 3D-lanepoint coordinates to 2D-imagepoint coordinates. If there's a huge hole in the list (None values),
        we need to split the list into two flat_lane_lists to make sure, the lanepoints are calculated and shown properly.
        
        Args:
            camera_rgb: carla.Actor. Get the camera_rgb actor to calculate the extrinsic matrix with the help of inverse matrix.
            lane_list: list. List of a lane, which contains its 3D-lanepoint coordinates x, y and z.

        Returns:
            List of 2D-points, where the elements of the list are tuples. Each tuple contains an x and y value.
        """
        flat_lane_list_a = []
        flat_lane_list_b = []
        lane_list = list(filter(lambda x: x is not None, lane_list))

        if lane_list:
            last_lanepoint = lane_list[0]

        for lanepoint in lane_list:
            if lanepoint and last_lanepoint:
                # Draw outer lanes not on junction
                distance = math.sqrt(math.pow(lanepoint.x - last_lanepoint.x, 2) +
                                     math.pow(lanepoint.y - last_lanepoint.y, 2) +
                                     math.pow(lanepoint.z - last_lanepoint.z, 2))

                # Check of there's a hole in the list
                if distance > cfg.meters_per_frame * 3:
                    flat_lane_list_b = flat_lane_list_a
                    flat_lane_list_a = []
                    last_lanepoint = lanepoint
                    continue

                last_lanepoint = lanepoint
                flat_lane_list_a.append([lanepoint.x, lanepoint.y, lanepoint.z, 1.0])

            else:
                # Just append a "Null" value
                flat_lane_list_a.append([None, None, None, None])

        if flat_lane_list_a:
            world_points = np.float32(flat_lane_list_a).T

            # This (4, 4) matrix transforms the points from world to sensor coordinates.
            world_2_camera = np.array(camera_rgb.get_transform().get_inverse_matrix())

            # Transform the points from world space to camera space.
            sensor_points = np.dot(world_2_camera, world_points)

            # Now we must change from UE4's coordinate system to a "standard" one
            # (x, y ,z) -> (y, -z, x)
            point_in_camera_coords = np.array([sensor_points[1],
                                               sensor_points[2] * -1,
                                               sensor_points[0]])

            # Finally we can use our intrinsic matrix to do the actual 3D -> 2D.
            points_2d = np.dot(self.camera_matrix, point_in_camera_coords)

            # Remember to normalize the x, y values by the 3rd value.
            points_2d = np.array([points_2d[0, :] / points_2d[2, :],
                                  points_2d[1, :] / points_2d[2, :],
                                  points_2d[2, :]])

            # visualize everything on a screen, the points that are out of the screen
            # must be discarted, the same with points behind the camera projection plane.
            points_2d = points_2d.T
            points_in_canvas_mask = (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < IMAGE_WIDTH) & \
                                    (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < IMAGE_HEIGHT) & \
                                    (points_2d[:, 2] > 0.0)

            points_2d = points_2d[points_in_canvas_mask]

            # Extract the screen coords (xy) as integers.
            x_coord = points_2d[:, 0].astype(np.int)
            y_coord = points_2d[:, 1].astype(np.int)
        else:
            x_coord = []
            y_coord = []

        if flat_lane_list_b:
            world_points = np.float32(flat_lane_list_b).T

            # This (4, 4) matrix transforms the points from world to sensor coordinates.
            world_2_camera = np.array(camera_rgb.get_transform().get_inverse_matrix())

            # Transform the points from world space to camera space.
            sensor_points = np.dot(world_2_camera, world_points)

            # Now we must change from UE4's coordinate system to a "standard" one
            # (x, y ,z) -> (y, -z, x)
            point_in_camera_coords = np.array([sensor_points[1],
                                               sensor_points[2] * -1,
                                               sensor_points[0]])

            # Finally we can use our intrinsic matrix to do the actual 3D -> 2D.
            points_2d = np.dot(self.camera_matrix, point_in_camera_coords)

            # Remember to normalize the x, y values by the 3rd value.
            points_2d = np.array([points_2d[0, :] / points_2d[2, :],
                                  points_2d[1, :] / points_2d[2, :],
                                  points_2d[2, :]])

            # visualize everything on a screen, the points that are out of the screen
            # must be discarted, the same with points behind the camera projection plane.
            points_2d = points_2d.T
            points_in_canvas_mask = (points_2d[:, 0] > 0.0) & (points_2d[:, 0] < IMAGE_WIDTH) & \
                                    (points_2d[:, 1] > 0.0) & (points_2d[:, 1] < IMAGE_HEIGHT) & \
                                    (points_2d[:, 2] > 0.0)

            points_2d = points_2d[points_in_canvas_mask]

            old_x_coord = np.insert(x_coord, 0, -1)
            old_y_coord = np.insert(y_coord, 0, -1)

            # Extract the screen coords (xy) as integers.
            new_x_coord = points_2d[:, 0].astype(np.int)
            new_y_coord = points_2d[:, 1].astype(np.int)

            x_coord = np.concatenate((new_x_coord, old_x_coord), axis=None)
            y_coord = np.concatenate((new_y_coord, old_y_coord), axis=None)

        return list(zip(x_coord, y_coord))

    @staticmethod
    def calculate_intersections(lane_list):
        """
        Transforms the 2D-image coordinates to the desired format, where only y-values in steps of 10 are needed.
        This is done by the intersection of the line of the two points enclosing the y-value and the line f(x) = y. 
        For the lower half (0.6) of the image, if there are no points enclosing the y-value, the intersection is calculate by the first points existing.
        This may leads to inaccurate results at junctions, but completes the lines until the bottom of the image. 
        Since junctions are excluded from trainingsdata the incorrect results at junctions are as well.
        
        Args:
            lane_list: list. List of a lane, which contains its 2D-image-coordinates x, y.

        Returns:
            List of 2D-points in the correct format for the deep learning algorithm, where the elements of the list 
            are tuples. Each tuple contains an x and y value.
        """
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
                for y_value in reversed(cfg.h_samples):
                    if gap and (last_point[1] >= y_value > xy_val[1]):
                        x_coord.append(-2)
                    elif (last_point == lane_list[0] and last_point[1] < y_value) and last_point[1] < IMAGE_HEIGHT - 0.5 * IMAGE_HEIGHT:
                        x_coord.append(-2)
                    elif (last_point[1] >= y_value > xy_val[1]) or (last_point == lane_list[0] and last_point[1] < y_value) and \
                            last_point[1] >= IMAGE_HEIGHT - 0.5 * IMAGE_HEIGHT:
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

            while len(x_coord) < len(cfg.h_samples):
                x_coord.append(-2)
            return list(zip(reversed(x_coord), cfg.h_samples))
        else:
            for _ in cfg.h_samples:
                x_coord.append(-2)
            return list(zip(x_coord, cfg.h_samples))

    @staticmethod
    def filter_2d_lanepoints(lane_list, image):
        """
        Remove all calculated 2D-lanepoints from the lane_list, which are e.g. on a house or wall, with the help of the sematic segmentation camera.
  
        Args:
            lane_list: list. List of a lane, which contains its lanepoints. Needed to check, if a lanepoint is located on a semantic tag like road or roadline.
            image: numpy array. Semantic segmentation image providing specific colorlabels to identify, if road or roadline.

        Returns:
            filtered_lane_list: list. Every point, which is located on a road or roadline, but not on a building or wall.
        """
        filtered_lane_list = []
        for lanepoint in lane_list:
            x = lanepoint[0]
            y = lanepoint[1]

            if (np.any(image[y][x] == (128, 64, 128), axis=-1) or  # Road
                    np.any(image[y][x] == (157, 234, 50), axis=-1) or  # Roadline
                    np.any(image[y][x] == (244, 35, 232), axis=-1) or  # Sidewalk
                    np.any(image[y][x] == (220, 220, 0), axis=-1) or  # Traffic sign
                    np.any(image[y][x] == (0, 0, 142), axis=-1)):  # Vehicle
                filtered_lane_list.append(lanepoint)
            else:
                filtered_lane_list.append((-2, y))
        return filtered_lane_list

    @staticmethod
    def format_2d_lanepoints(lane_list):
        """
        Args: 
            lane_list: list. List of a lane, which contains its lanepoints. 

        Returns:
            x_lane_list: list of x-coordinates. Extracts all the x values from the lane_list.
            Formats the lane_list as followed: [(x0,y0),(x1,y1),...,(xn,yn)] to [x0, x1, ..., xn]
        """
        x_lane_list = []
        for lanepoint in lane_list:
            x_lane_list.append(lanepoint[0])

        return x_lane_list


# ==============================================================================
# -- 5. Main -------------------------------------------------------------------
# ==============================================================================


if __name__ == '__main__':
    try:
        CarlaGame().execute()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
