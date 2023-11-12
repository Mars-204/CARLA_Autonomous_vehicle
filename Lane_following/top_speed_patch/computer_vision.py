
# add your global imports here
import cv2
import numpy as np
import pygame
import random
import math

global topspeed
topspeed = 0

"""
We expect from you to implement a run_step() function that will return data required by the Driver to steer the car.
Our example uses RGB camera and OpenCV, but you can come up with any approach you wish
"""

class Navigator(object):

    def __init__(self, world):
        self.world = world
        self.lidar = world.lidar_manager
        self.camera_rgb = world.camera_manager
        pass
    

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
        control_data["target_speed"] = random.randint(0, 50)
        control_data["curve"] = random.randint(-90, 90)
        
        return control_data

