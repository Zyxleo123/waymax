"""Stand-in for nuplan-devkit's vehicle_parameters, the only nuPlan import on Diffusion Planner's training path
(utils/data_augmentation.py reads the wheel base). Values are nuPlan's Pacifica parameters."""

from types import SimpleNamespace


def get_pacifica_parameters():
    return SimpleNamespace(
        vehicle_name="pacifica",
        vehicle_type="gen1",
        width=1.1485 * 2.0,
        front_length=4.049,
        rear_length=1.127,
        wheel_base=3.089,
        cog_position_from_rear_axle=1.67,
        height=1.777,
    )
