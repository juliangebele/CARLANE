IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 20
FOV = 90.0

# Recommended maps:
# With worldobjects: 'Town04', 'Town06'
# Without worldobjects: 'Town04_Opt', 'Town06_Opt'
CARLA_TOWN = 'Town06'

# Save images and labels
is_saving = True
# Draw all 3D lanes in carla simulator
draw_lanes = False
# Calculate and draw 3D lanes on junction
junction_mode = False
# Third-person view for the ego vehicle
is_third_person = False
# Only use the inner two lanes
is_solo_lane = False

# Angles, that determine the rotation of the own car
rotation_angles = [-2, 0, 2]
# Use the rotation from the last nth waypoint
number_of_points_visited = 14 if CARLA_TOWN == 'Town10HD' or CARLA_TOWN == 'Town10HD_Opt' else 5
# Max size per imageclass (steep_left, left, straight, right or steep_right)
max_images_per_class = 200
# Number of images stored in the image buffer
number_of_images = 200
# Distance between the calculated lanepoints
meters_per_frame = 1.0
# Total length of a lane_list
number_of_lanepoints = 60
# Vertical startposition of the lanepoints in the 2D-image
row_anchor_start = 160
# Vertical row anchors
h_samples = [y for y in range(row_anchor_start, IMAGE_HEIGHT, 10)]
# Reset interval in ticks
reset_intervall = 200
# Lane angle to classify lanes [5, 15] -> angle < 5 deg is straight, angle < 15 is curve
lane_angle = {'straight': 5, 'curve': 12}
# Distance in meters from ego vehicle to classify lanes based on their angle
waypoint_distance = 40

# Root directory of the dataset
root = 'TuLane/data/carla/'
# Directory, where the images are saved
suffix = 'val/sim/' + CARLA_TOWN + '/'
# Saving directory of the collected data
saving_directory = root + suffix
