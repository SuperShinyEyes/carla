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

import queue

try:
    sys.path.append(
        glob.glob(
            "../carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla
from carla import ColorConverter as cc

Attachment = carla.AttachmentType

from typing import *
import time
import logging
import random
from pathlib import Path
from datetime import datetime
import random; random.seed(1)

import numpy as np
from numpy import ndarray

from cv2 import VideoWriter_fourcc
import cv2 as cv
import tqdm

import util

images = []
WIDTH, HEIGHT = 1280, 720

def get_vehicle(world):
    bp = world.get_blueprint_library().filter("model3")[0]
    print(bp)
    bp.set_attribute("role_name", "vehicle")
    if bp.has_attribute("is_invincible"):
        bp.set_attribute("is_invincible", "true")
    # Spawn the player.
    spawn_points = world.get_map().get_spawn_points()
    # spawn_point = random.choice(spawn_points)
    spawn_point = spawn_points[0]
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


def get_camera(world, parent_actor, height, width, gamma):
    """Configure camera just like Openpilot 0.8.8
    - https://github.com/SuperShinyEyes/openpilot/blob/204e5a090735a059d69c29145a4cee49450da07e/tools/sim/bridge.py#L194-L198
    - https://carla.readthedocs.io/en/0.9.11/ref_sensors/#rgb-camera
    - https://carla.readthedocs.io/en/0.9.11/ref_sensors/#advanced-camera-attributes
    """
    bp_library = world.get_blueprint_library()
    bp = bp_library.find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(width))
    bp.set_attribute("image_size_y", str(height))
    bp.set_attribute("fov", "70")
    bp.set_attribute("shutter_speed", "60")
    bp.set_attribute("chromatic_aberration_intensity", "0.5")
    bp.set_attribute("motion_blur_intensity", "0.45")
    # bp.set_attribute('sensor_tick', '0.033')

    if bp.has_attribute("gamma"):
        bp.set_attribute("gamma", str(gamma))

    camera = world.spawn_actor(
        bp,
        carla.Transform(carla.Location(x=1.6, z=1.7)),
        attach_to=parent_actor,
        # attachment=Attachment.Rigid,
    )
    return camera


def set_target_velocity(
    vehicle: "carla.Actor", velocity_in_kph: "carla.libcarla.Vector3D"
):
    velocity_in_mps = carla.libcarla.Vector3D(
        velocity_in_kph.x / 3.6,
        velocity_in_kph.y / 3.6,
        velocity_in_kph.z / 3.6,
    )

    vehicle.set_target_velocity(velocity_in_mps)


def carla_camera_data_to_numpy(data) -> ndarray:
    data.convert(cc.Raw)
    image = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    image = np.reshape(image, (data.height, data.width, 4))
    image = image[:, :, :3]
    return image


def save_video(camera_data: List, video_path: str, fps: int, resolution: Tuple[int, int]):
    vwriter = cv.VideoWriter(
        video_path, VideoWriter_fourcc(*"mp4v"), fps, resolution
    )

    for data in tqdm.tqdm(camera_data):
        vwriter.write(carla_camera_data_to_numpy(data))

    vwriter.release()

class CarlaSyncMode(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CarlaSyncMode(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    """

    def __init__(self, world, camera, fps):
        self.world = world
        self.camera = camera
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._queues = []
        self._settings = None
        self._world_tick_id = None

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=self.delta_seconds)
        )

        q = queue.Queue()
        self._world_tick_id = self.world.on_tick(q.put)
        self._queues.append(q)

        q = queue.Queue()
        self.camera.listen(q.put)
        self._queues.append(q)

        return self

    def tick(self, timeout):
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def __exit__(self, *args, **kwargs):
        """
        
        About deregistering world tick: https://github.com/carla-simulator/carla/pull/1865
        """
        assert self._world_tick_id is not None
        self.world.remove_on_tick(self._world_tick_id)
        self.world.apply_settings(self._settings)

        self.camera.stop()

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data


class Recorder(object):
    def __init__(self, fps, duration_in_second, height, width, gamma):
        self.camera_data = []
        self.vehicle = None
        self.fps: int = fps
        self.resolution = (width, height)  # OpenCV VideoWriter format
        self.duration_in_second = duration_in_second
        self.target_n_frames = fps * duration_in_second

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(3.0)
        self.world = self.client.get_world()
        self.vehicle_velocity = carla.libcarla.Vector3D(-60, 0, 0)
        self.vehicle = get_vehicle(self.world)

        self.camera = get_camera(
            self.world,
            parent_actor=self.vehicle,
            height=height,
            width=width,
            gamma=gamma,
        )
        logging.info(
            f"Capture video; fps={fps}, duration_in_second={duration_in_second}, resolution={width}x{height}"
        )

    def _set_world_setting(self, fps):
        """
        - [A question on fixed_delta_seconds of synchronous_mode](https://github.com/carla-simulator/carla/issues/3479)
        """
        settings = self.world.get_settings()
        # settings.fixed_delta_seconds = (1.0 / fps)
        self.world.apply_settings(settings)

    def _warmup(self, seconds=2):
        logging.info(f"Warm up for {seconds} seconds.")
        time.sleep(seconds)

    def reset_camera(self, yaw: float, pitch: float):
        pass

    def reset_car(self):
        set_target_velocity(self.vehicle, self.vehicle_velocity)

    def record(self, output_path: str, yaw: float, pitch: float):
        self.reset_car()
        self._warmup(seconds=2)
        
        ############################
        # Capture video
        print(f"Simulating {output_path}")
        _start = time.perf_counter()
        
        with CarlaSyncMode(self.world, self.camera, fps=self.fps) as sync_mode:
            for _ in range(self.target_n_frames):
                # Advance the simulation and wait for the data.
                _, camera_data = sync_mode.tick(timeout=2.0)
                self.camera_data.append(camera_data)

        _end = time.perf_counter()
        print(
            f"{self.duration_in_second}-second simulation took {_end - _start:.2f} seconds."
        )

        ############################

        save_video(self.camera_data, output_path, self.fps, self.resolution)
        self.camera_data.clear()

    def destroy(self):
        logging.info("Destroy the cars and cameras")
        util.destroy_vehicles(self.world)
        util.destroy_sensors(self.world)
        

def _make_dataset_directory() -> Path:
    path = "camera_yaw_pitch_dataset-{0:%Y_%m_%d-%H_%M_%S}".format(datetime.now())
    path = Path(path)
    path.mkdir()
    return path

def get_random_angle() -> float:
    """Return a random angle (in degree) in the range [-20, +20] rounded by second decimal."""
    return round(random.uniform(-20,20), 2)

def main(n_video: int, height, width, fps, duration_in_second, gamma=2.2):

    dataset_path: Path = _make_dataset_directory()
    video_name_template = "{index:03d}-pitch={pitch}-yaw={yaw}.mp4"

    recorder = Recorder(
        fps=fps,
        duration_in_second=duration_in_second,
        height=height,
        width=width,
        gamma=gamma,
    )

    for i in range(n_video):
        pitch: float = get_random_angle()
        yaw: float = get_random_angle()
        video_name = video_name_template.format(index=i, pitch=pitch, yaw=yaw)
        video_path: str = str(dataset_path / video_name)

        assert not os.path.exists(video_path), f'{video_path} exists already!'

        try:
            recorder.record(video_path, yaw=yaw, pitch=pitch)
        except Exception as e:
            print(e)
            recorder.destroy()
            return
        finally:
            pass


if __name__ == "__main__":
    log_level = logging.DEBUG if True else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    main(
        n_video=3,
        fps=30,
        duration_in_second=5,
        height=720,
        width=1280,
        gamma=2.2
    )

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
    # recorder = Recorder(map='/Game/Carla/Maps/Town05',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town02_Opt',)
    # recorder = Recorder(map='/Game/Carla/Maps/Town03')
    # recorder = Recorder(map='/Game/Carla/Maps/Town01',)