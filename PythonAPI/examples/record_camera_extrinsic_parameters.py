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
RECORD_LENGTH_IN_SEC = 5

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

    return vehicle

def get_camera(world, parent_actor, height, width, gamma):
    bp_library = world.get_blueprint_library()
    item = ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}]
    bp = bp_library.find(item[0])
    # camera_bp = world.get_blueprint_library().find('sensor.camera.depth')
    bp.set_attribute('image_size_x', str(width))
    bp.set_attribute('image_size_y', str(height))
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

def save_video(images, filepath):
    
    resolution = (WIDTH, HEIGHT)
    fps_real = len(images)/RECORD_LENGTH_IN_SEC
    fps = int(fps_real)
    print(f"Make video from {len(images)} frames; {fps_real} fps")
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
        
        self.client.load_world(map)
        self.client.reload_world()

        self.world = self.client.get_world()

        # Enable autopilot
        traffic_manager = self.client.get_trafficmanager(8000)
        # traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        # traffic_manager.ignore_lights_percentage(vehicle,100)
        traffic_manager.set_hybrid_physics_mode(True)

        self.vehicle = get_vehicle(self.world)
        self.vehicle.set_autopilot(True)

    def record(self, filepath, duration_in_second, height, width, gamma):
        camera = get_camera(self.world, parent_actor=self.vehicle, height=height, width=width, gamma=gamma)
        print(camera)

        time.sleep(1)   # Wait until autopilot kicks in properly
        weak_self = weakref.ref(self)
        camera.listen(lambda data: Recorder.buffer_camera_data(weak_self, data))
        time.sleep(duration_in_second)

        if camera is not None:
            camera.destroy()

        save_video(images=self.images, filepath=filepath)
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
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.model3")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
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
    recorder = Recorder(map='/Game/Carla/Maps/Town07_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town06',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town01_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town01',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town05',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town05_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town02_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town03')
    recorder.record(filepath='output.mp4', duration_in_second=5, height=args.height, width=args.width, gamma=args.gamma,)
        

if __name__ == '__main__':

    main()
