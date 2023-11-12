
# add your global imports here
import cv2
import numpy as np
import pygame
import random
import carla


"""
We expect from you to implement a run_step() function that will return data required by the Driver to steer the car.
Our example uses RGB camera and OpenCV, but you can come up with any approach you wish
"""
def show_speed(self, img):
    global topspeed
    h, w = img.shape[:2]
    v = self.world.player.get_velocity()
    speed = round((3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),2)
    topspeed = speed if topspeed < speed else topspeed 
    square_img = np.zeros_like(img, np.uint8)
    square_img = cv2.rectangle(square_img, (40, 30), (300, 80), (255,255,255), cv2.FILLED)
    img = cv2.addWeighted(img, 1, square_img, 0.2, 1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5 
    color = (0, 0, 0)
    thickness = 1
    line1=f'Speed: {speed} km/h'
    org1 = (50, 50)
    line2=f'Top speed: {topspeed} km/h'
    org2 = (50, 70)
    img = cv2.putText(img, line1, org1, font, 
               fontScale, color, thickness, cv2.LINE_AA)
    img = cv2.putText(img, line2, org2, font, 
               fontScale, color, thickness, cv2.LINE_AA)
    
    return img


class Navigator(object):

    def __init__(self, world):
        self.world = world
        self.lidar = world.lidar_manager
        self.camera_rgb = world.camera_manager
        pass

    def run_step(self):
        """
        Method analyzing the sensors to obtain the data needed for car steering
        
        Returns:
            control_data: dict
        """

        control_data = dict()

        """
        The following lines will provide you with the image data from Camera Sensor
        - You can select position of the sensor manually in game mode before initiating the automatic control
        - pygame library provides a surface class that can be used to get 3-D image array to be later used in
            opencv
        - However the surface array has swapped axes, so it cannot be used directly in opencv
        """
        cv2.imshow("OpenCV camera view", self.show_speed(self.camera_rgb.image))
        cv2.imshow("lidar", self.lidar.image)
        cv2.waitKey(1)
        
        
        """
        Here, you can see exemplary data you can return from this step. It should help the driver
        fulfill the tasks
        """
        control_data["target_speed"], control_data["curve"] = self.process_image(self.camera_rgb.image)
        
        return control_data

    def calculate_curvature(self, edges):
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

    def apply_lane_following_control(self, edges):
        # Determine the lane boundaries based on Canny edge detection
        # Constants for control logic
        TARGET_SPEED = 10  # Target speed in m/s
        TARGET_LANE_CENTER = 320  # Target lane center (assuming 640x480 image)
        CURVATURE_GAIN = 1  # Gain for proportional control based on curvature
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
            curvature = self.calculate_curvature(edges)
            throttle = min(1.0, TARGET_SPEED / (self.world.vehicle.get_velocity().x * (1 + abs(CURVATURE_GAIN*curvature))))

            ## Throttle control to maintain the target speed
            # throttle = min(1.0, TARGET_SPEED / vehicle.get_velocity().x)

            # Apply control to the vehicle

            return throttle, steer

        return carla.VehicleControl()

    def process_image(self, image):
        height, width = 720 , 1280   # The camera image is 720x1280x3
        img_array = image  # image is already a numpy array. Else we can convert to numpy array
        img = img_array.reshape(img_array,(height,width,4))

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


        # Apply control logic to stay within the lanes and maintain speed
        target_speed, steer = self.apply_lane_following_control(masked_edges)

        return target_speed, steer






