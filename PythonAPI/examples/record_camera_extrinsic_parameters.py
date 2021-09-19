#!/usr/bin/env python

"""
Record a video file
"""
from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
from carla import ColorConverter as cc
Attachment = carla.AttachmentType

import weakref
import time
import argparse
import logging
import random

import numpy as np
from cv2 import (CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT,
                 CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH,
                 CAP_PROP_POS_FRAMES, VideoWriter_fourcc)
import cv2 as cv

images = []
WIDTH, HEIGHT = 1280, 720
RECORD_LENGTH_IN_SEC = 60

def get_vehicle(world):
    bp = world.get_blueprint_library().filter('model3')[0]
    print(bp)
    bp.set_attribute('role_name', 'vehicle')
    if bp.has_attribute('is_invincible'):
        bp.set_attribute('is_invincible', 'true')
    # Spawn the player.
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points) 
    vehicle = world.spawn_actor(bp, spawn_point)
    
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
    
    # physics_control = vehicle.get_physics_control()
    # physics_control.use_sweep_wheel_collision = True
    # vehicle.apply_physics_control(physics_control)

    
    # Vehicle physics setting from Openpilot make tires less slippery
    # wheel_control = carla.WheelPhysicsControl(tire_friction=5)
    physics_control = vehicle.get_physics_control()
    physics_control.mass = 2326
    # physics_control.wheels = [wheel_control]*4
    physics_control.torque_curve = [[20.0, 500.0], [5000.0, 500.0]]
    physics_control.gear_switch_time = 0.0
    vehicle.apply_physics_control(physics_control)


    return vehicle

def get_camera(world, parent_actor, height, width, gamma, fps):
    bp_library = world.get_blueprint_library()
    item = ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}]
    bp = bp_library.find(item[0])
    # camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
    bp.set_attribute('image_size_x', str(width))
    bp.set_attribute('image_size_y', str(height))
    bp.set_attribute('fov', '70')
    bp.set_attribute('sensor_tick', str(1/fps))
  
    if bp.has_attribute('gamma'):
        bp.set_attribute('gamma', str(gamma))
    for attr_name, attr_value in item[3].items():
        bp.set_attribute(attr_name, attr_value)

    camera = world.spawn_actor(
        bp,
        carla.Transform(carla.Location(x=1.6, z=1.7)),
        attach_to=parent_actor,
        # attachment=Attachment.Rigid,
    )
    print("Attached camera")
    return camera

def carla_image_to_numpy(data):
    data.convert(cc.Raw)
    image = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    image = np.reshape(image, (data.height, data.width, 4))
    image = image[:, :, :3]
    return image

def save_video(images, filepath, fps):
    
    resolution = (WIDTH, HEIGHT)
    vwriter = cv.VideoWriter(filepath, VideoWriter_fourcc(*'mp4v'), fps,
                            resolution)
    for data in images:
        image = carla_image_to_numpy(data)
        vwriter.write(image)

    vwriter.release()
class Recorder(object):

    def __init__(self, map):
        self.images = []
        self.vehicle = None
        
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(3.0)
        
        # self.client.load_world(map)
        # self.client.reload_world()

        self.world = self.client.get_world()

        # Enable autopilot
        traffic_manager = self.client.get_trafficmanager(8000)
        # traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        # traffic_manager.ignore_lights_percentage(vehicle,100)
        traffic_manager.set_hybrid_physics_mode(True)

        self.vehicle = get_vehicle(self.world)
        self.vehicle.set_autopilot(True)

    def _set_world_setting(self, fps):
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = (1.0 / fps)
        self.world.apply_settings(settings)


    def record(self, filepath, fps, duration_in_second, height, width, gamma):
        self._set_world_setting(fps)
        max_n_frame = fps * duration_in_second
        
        camera = get_camera(self.world, parent_actor=self.vehicle, height=height, width=width, gamma=gamma, fps=fps)
        print(camera)
        logging.info(f"Capture video; fps={fps}, duration_in_second={duration_in_second}, resolution={width}x{height}")
        ############################
        # Capture video
        time.sleep(1)   # Wait until autopilot kicks in properly
        _start = time.perf_counter()
        weak_self = weakref.ref(self)
        camera.listen(lambda data: Recorder.buffer_camera_data(weak_self, data))

        while 1:
            if len(self.images) >= max_n_frame:
                break
            time.sleep(0.1)

        _end = time.perf_counter()
        print(f"Simulation of {duration_in_second} seconds took {_end - _start:.2f} seconds.")
        if camera is not None:
            camera.destroy()

        ############################

        save_video(images=self.images, filepath=filepath, fps=fps)
        self.images = []

    def __del__(self, *args):
        print("Destroy the car")
        if self.vehicle is not None:
            carla.command.DestroyActor(self.vehicle)

    @staticmethod
    def buffer_camera_data(weak_self, data):
        self = weak_self()
        if not self: 
            print(f'self is None: {self}')
            return

        self.images.append(data)
        

def main():
    log_level = logging.DEBUG if True else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    print(__doc__)

    # recorder = Recorder(map='/Game/Carla/Maps/Town10HD',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town04_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town10HD_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town07',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town04',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town02',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town06_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town03_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town07_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town06',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town01_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town01',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town05',)
    recorder = Recorder(map='/Game/Carla/Maps/Town05_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town02_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town03')
    recorder.record(filepath='output.mp4', fps=30, duration_in_second=60, height=720, width=1280, gamma=2.2,)
        

if __name__ == '__main__':

    main()
