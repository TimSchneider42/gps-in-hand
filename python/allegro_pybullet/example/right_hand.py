"""
In this example the right hand of the Allegro robot is controlled with random signals in torque mode.
"""

from time import sleep

import math
import numpy as np
import pybullet
import pybullet_data
from numpy import random

from allegro_pybullet import PhysicsClient
from allegro_pybullet.simulation_body import URDFBody
from allegro_pybullet.simulation_body.allegro_hand import AllegroFingerTypes, AllegroFingerJointTypes, AllegroRightHand
from allegro_pybullet.simulation_object import JointControlMode

# Create PhysicsClient instance and connect it to a visual physics simulation. If no GUI is required, use
# connect_direct. "connect" is a bit misleading here as this is actually creating a new simulation.
# PhysicsClient is essentially a wrapper around pybullet, which prevents you from using their awful API too much.
pc = PhysicsClient()
pc.connect_gui()

# Some general simulation settings
# Set the time step to 2ms
pc.time_step = 0.002
# Configure the camera of the GUI
pc.reset_debug_visualizer_camera(0.7, 0, -45, np.array([0, 0, 0]))
# Disable mouse picking as we do not need it
pc.configure_debug_visualizer(pybullet.COV_ENABLE_MOUSE_PICKING, False)

# Create a AllegroRightHand body and specify the position. Please note every body needs to be added to the simulation
# before it can be used.
hand = AllegroRightHand(np.array([0.2, 0.0, 0.2]), np.array(pybullet.getQuaternionFromEuler([math.pi / 2, 0, math.pi])),
                        joint_control_mode=JointControlMode.TORQUE_CONTROL)
# Add the hand to the simulation. Now it can be used.
pc.add_body(hand)

# Add a plane from the standard pybullet data to the simulation.
pc.set_additional_search_path(pybullet_data.getDataPath())
# URDFBody lets you load arbitrary urdf models and provides access to all their joints and links.
pc.add_body(URDFBody("plane.urdf"))

# Now we are going to simulate 10 seconds and apply random torques to all joints of the hand
for i in range(5000):
    for ft in AllegroFingerTypes:
        for jt in AllegroFingerJointTypes:
            hand.fingers[ft].joints[jt].torque = random.uniform(-0.1, 0.1)
    pc.step_simulation()
    sleep(pc.time_step)
