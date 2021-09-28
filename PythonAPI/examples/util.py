import carla
from carla import ColorConverter as cc
import numpy as np
import math


def carla_image_to_numpy(data):
    data.convert(cc.Raw)
    image = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
    image = np.reshape(image, (data.height, data.width, 4))
    image = image[:, :, :3]
    return image


def get_linear_velocity_in_kph(v: "carla.libcarla.Vector3D") -> float:
    return 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)


def destroy_vehicles(world: "carla.libcarla.World"):
    for actor in world.get_actors().filter("vehicle.*"):
        actor.destroy()

def destroy_sensors(world: "carla.libcarla.World"):
    for actor in world.get_actors().filter("sensor.*"):
        actor.destroy()

def print_all_actors(world: "carla.libcarla.World"):
    for actor in world.get_actors():
        print(actor)

def set_target_velocity(
    vehicle: "carla.Actor.", velocity_in_kph: "carla.libcarla.Vector3D"
):
    velocity_in_mps = carla.libcarla.Vector3D(
        velocity_in_kph.x / 3.6,
        velocity_in_kph.y / 3.6,
        velocity_in_kph.z / 3.6,
    )

    vehicle.set_target_velocity(velocity_in_mps)
