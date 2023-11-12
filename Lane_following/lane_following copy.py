import carla
import pygame
import numpy as np
import cv2
import random
import time

# Connect to the Carla server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the Carla world and blueprint library
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Spawn a vehicle
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Spawn a camera sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '110')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Lane Detection")

clock = pygame.time.Clock()

# Constants for control logic
TARGET_SPEED = 10  # Target speed in m/s
TARGET_LANE_CENTER = 320  # Target lane center (assuming 640x480 image)
CURVATURE_GAIN = 1  # Gain for proportional control based on curvature

def calculate_curvature(edges):
    # Calculate the curvature of the road based on Canny edge detection
    lane_boundaries = np.argwhere(edges > 0)

    if lane_boundaries.size > 0:
        y, x = lane_boundaries[:, 0], lane_boundaries[:, 1]
        # Fit a second-degree polynomial to the lane boundaries
        curve_fit = np.polyfit(y, x, 2)
        curvature = 2 * curve_fit[0] * np.max(y)  # Curvature at the bottom of the image
        return curvature
    else:
        return 0.0

def apply_lane_following_control(edges):
    # Determine the lane boundaries based on Canny edge detection
    lane_boundaries = np.argwhere(edges > 0)
    #PID control
    KP=0.01
    KI=0.01
    KD=0.1

    integral =0
    prev_error=0

    if lane_boundaries.size > 0:
        # Calculate the center of the detected lanes
        lane_center = float(np.mean(lane_boundaries[:, 1]))

        # Calculate the error from the target lane center
        error = TARGET_LANE_CENTER - lane_center

        #PID control adjust
        proportional = KP*error
        integral=integral+(KI*error)
        derivative =KD * (error-prev_error)
        #Steering adjustment
        steer = proportional+integral+derivative

        #Update error
        prev_error =error

        # Proportional control to adjust steering based on the error
        steer = max(-1.0,min(1.0,steer))

        # Throttle control to maintain the target speed with curvature consideration
        curvature = calculate_curvature(edges)
        throttle = min(1.0, TARGET_SPEED / (vehicle.get_velocity().x * (1 + abs(CURVATURE_GAIN*curvature))))

        ## Throttle control to maintain the target speed
        # throttle = min(1.0, TARGET_SPEED / vehicle.get_velocity().x)

        # Apply control to the vehicle
        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = 0.0
        return control

    return carla.VehicleControl()

def process_image(image):
    # Convert the raw image data to a numpy array
    #img_array = np.array(image.raw_data)
    img_array = np.frombuffer(image.raw_data,dtype=np.dtype('uint8'))
    height,width = image.height, image.width
    img = img_array.reshape(img_array,(height,width,4))

    #height ,width, _ = img_array.shape
    img = img_array[height //2: ,:,:]
    

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(img, 50, 150)

     # Region of Interest (ROI) mask to focus on the road lanes
    mask = np.zeros_like(edges)
    height, width = edges.shape
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    cv2.fillPoly(mask, [np.array(roi_vertices, np.int32)], 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Rotate the images by 90 degrees
    rotated_gray = pygame.surfarray.make_surface(np.rot90(gray))
    rotated_edges = pygame.surfarray.make_surface(np.rot90(edges))

    # Apply control logic to stay within the lanes and maintain speed
    control = apply_lane_following_control(masked_edges)

    # Apply control to the vehicle
    vehicle.apply_control(control)

    # Display the rotated images with Pygame
    screen.blit(rotated_gray, (0, 0))
    # screen.blit(rotated_edges, (0, 240))
    pygame.display.flip()

# Register the callback function for the camera sensor
camera.listen(lambda image: process_image(image))

try:
    # Let the simulation run for a certain duration
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Control Pygame frame rate
        clock.tick(60)

finally:
    # Cleanup: Destroy the vehicle and camera
    vehicle.destroy()
    camera.destroy()
    pygame.quit()
