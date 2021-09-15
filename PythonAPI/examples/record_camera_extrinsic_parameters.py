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
# self._camera_transforms = [
#             (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
#             (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
#             (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
#             (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
#             (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]
#         self.transform_index = 1
#         self.sensors = [
#             ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
#             ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
#             ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
#             ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
#             ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
#             ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
#                 'Camera Semantic Segmentation (CityScapes Palette)', {}],
#             ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50'}],
#             ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
#             ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
#                 {'lens_circle_multiplier': '3.0',
#                 'lens_circle_falloff': '3.0',
#                 'chromatic_aberration_intensity': '0.5',
#                 'chromatic_aberration_offset': '0'}]]

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
    
    vehicle.set_autopilot(True)
    # vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
    
    physics_control = vehicle.get_physics_control()
    physics_control.use_sweep_wheel_collision = True
    vehicle.apply_physics_control(physics_control)

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

    
def image_to_buffer(data):
    global images
    image = np.array(data.raw_data)
    image = image.reshape((HEIGHT, WIDTH, 4))
    image = image[:,:,:3]
    images.append(image)
    print(f"Saved {len(images)} images")

def save_video(images):
    
    resolution = (WIDTH, HEIGHT)
    print(f"Make video from {len(images)} frames")
    vwriter = cv.VideoWriter('out.mp4', VideoWriter_fourcc(*'MP4V'), 30,
                              resolution)
    for image in images:
        vwriter.write(image)

    vwriter.release()


def loop(args):
    global images
    world = camera = None
    actors = []
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        world = client.get_world()

        # Enable autopilot
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.set_hybrid_physics_mode(True)

        for _ in range(1):
            vehicle = get_vehicle(world)
            actors.append(vehicle)

        camera = get_camera(world, parent_actor=vehicle, height=args.height, width=args.width, gamma=args.gamma)
        print(camera)
        # camera.listen(lambda image: image.save_to_disk('_out/%06d.png' % image.frame, cc))
        camera.listen(lambda data: image_to_buffer(data))

        time.sleep(5)
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(e)
        print(repr(e))
        print(e.args)
    finally:
        save_video(images=images)
        for actor in actors:
            actor.destroy()
        if camera is not None:
            camera.destroy()

        # if world is not None:
        #     world.destroy()


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

    try:

        loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
