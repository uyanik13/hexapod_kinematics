import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from core.kinematic.leg_kinematics import Kinematics
import logging


def test_leg_positions():
    """Test different leg positions and plot them."""
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 5))

    # Test positions (coxa, femur, tibia angles in degrees)
    test_configs = [
        ([0, 0, 0], "Home Position\n[0,0,0]"),
        ([0, 30, 90], "Raised Position\n[0,45,90]"),
        ([-45, 30, 90], "Forward Position\n[-45,45,90]"),
        ([45, 30, 90], "Backward Position\n[45,45,90]"),
    ]

    # Test each configuration
    for i, (angles, title) in enumerate(test_configs):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")  # 1x4 grid, i+1'inci subplot
        create_and_plot_leg(ax, angles, title)

    plt.tight_layout()
    plt.show()


def test_forward_kinematics():
    """Test forward kinematics calculations."""
    leg = Kinematics("test_leg", COXA=4.35, FEMUR=8.0, TIBIA=15.5)

    print("\nTesting Forward Kinematics:")
    test_configs = [
        [0, 0, 0],  # Home
        [0, 30, 90],  # Raised
        [-45, 30, 90],  # Forward
        [45, 30, 90],  # Backward
    ]

    for angles in test_configs:
        leg.set_angles(angles)
        pos = leg.get_current_position()
        print(f"\nAngles: {angles}")
        print(f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")


def test_inverse_kinematics():
    """Test inverse kinematics calculations."""
    leg = Kinematics("test_leg", COXA=4.35, FEMUR=8.0, TIBIA=15.5)

    print("\nTesting Inverse Kinematics:")

    # Test configurations
    test_configs = [
        ("Home Position", [0, 0, 0]),
        ("Raised Position", [0, 30, 90]),
        ("Forward Position", [-45, 30, 90]),
        ("Backward Position", [45, 30, 90]),
    ]

    for name, original_angles in test_configs:
        print(f"\nTesting {name}:")
        print(f"Original angles: {original_angles}")

        # Forward kinematics to get target position
        leg.set_angles(original_angles)
        target = leg.get_current_position()
        print(f"Target position: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")

        # Inverse kinematics to get angles
        calculated_angles = leg.inverse_kinematics(target)
        if calculated_angles is not None:
            print(f"Calculated angles: {[round(a, 1) for a in calculated_angles]}")

            # Calculate error
            angle_error = np.array(calculated_angles) - np.array(original_angles)
            print(f"Angle error: {[round(e, 1) for e in angle_error]}")
        else:
            print("Failed to find inverse kinematics solution")


def create_and_plot_leg(ax, angles, title):
    """Create a leg with given angles and plot it."""
    leg = Kinematics("test_leg", COXA=4.35, FEMUR=8.0, TIBIA=15.5)
    leg.set_angles(angles)
    leg.plot(ax)

    # Set plot limits and labels
    ax.set_xlim([-20, 20])  # Daha yakın görünüm için limitleri küçülttüm
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])
    ax.set_title(title)


def test_fk_ik_consistency():
    """Test consistency between forward and inverse kinematics."""
    leg = Kinematics("test_leg", COXA=4.35, FEMUR=8.0, TIBIA=15.5)

    print("\nTesting FK -> IK Consistency:")

    test_configs = [
        [0, 0, 0],  # Home
        [0, 30, 90],  # Raised
        [-45, 30, 90],  # Forward
        [45, 30, 90],  # Backward
    ]

    for original_angles in test_configs:
        print(f"\nOriginal angles: {original_angles}")

        # Forward kinematics to get position
        leg.set_angles(original_angles)
        position = leg.get_current_position()
        print(f"FK Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")

        # Inverse kinematics to get angles back
        calculated_angles = leg.inverse_kinematics(position)
        print(f"IK Angles: {[round(a, 2) for a in calculated_angles]}")

        # Calculate error
        if calculated_angles is not None:
            angle_error = np.array(calculated_angles) - np.array(original_angles)
            print(f"Angle error: {[round(e, 2) for e in angle_error]}")

            # Test position consistency
            leg.set_angles(calculated_angles)
            new_position = leg.get_current_position()
            position_error = np.linalg.norm(new_position - position)
            print(f"Position error: {position_error:.4f} cm")


if __name__ == "__main__":
    print("Testing leg positions...")
    test_leg_positions()
    # test_forward_kinematics()
    # test_inverse_kinematics()
    # test_fk_ik_consistency()
