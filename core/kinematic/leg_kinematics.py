import numpy as np
import roboticstoolbox as rtb
import logging
from math import degrees, radians, cos, sin, sqrt, acos, atan2, pi


class Kinematics:
    def __init__(self, name, COXA, FEMUR, TIBIA, parts=None):
        self.name = name
        self.COXA = float(COXA)  # 4.35cm
        self.FEMUR = float(FEMUR)  # 8.0cm
        self.TIBIA = float(TIBIA)  # 15.5cm

        # Body dimensions
        self.BODY_LENGTH = 11.0  # 11 cm (short edge)
        self.BODY_WIDTH = 15.0  # 15 cm (long edge)

        # Leg mounting points (relative to body center)
        self.mount_positions = {
            "LEG1": [
                self.BODY_LENGTH / 2 - 5.25 / 2,
                self.BODY_WIDTH / 2,
                0,
            ],  # Right front (1)
            "LEG2": [
                -self.BODY_LENGTH / 2 + 5.25 / 2,
                self.BODY_WIDTH / 2,
                0,
            ],  # Left front (2)
            "LEG3": [-self.BODY_LENGTH / 2 + 5.25 / 2, 0, 0],  # Left middle (3)
            "LEG4": [
                -self.BODY_LENGTH / 2 + 5.25 / 2,
                -self.BODY_WIDTH / 2,
                0,
            ],  # Left rear (4)
            "LEG5": [
                self.BODY_LENGTH / 2 - 5.25 / 2,
                -self.BODY_WIDTH / 2,
                0,
            ],  # Right rear (5)
            "LEG6": [self.BODY_LENGTH / 2 - 5.25 / 2, 0, 0],  # Right middle (6)
        }

        # Active leg (default is LEG1)
        self.active_leg = "LEG1"

        # Maximum reach distance - 95% of total length
        total_length = self.COXA + self.FEMUR + self.TIBIA
        self.MAX_REACH = total_length * 0.95
        # Minimum reach distance - 1.5 times COXA length
        self.MIN_REACH = self.COXA * 1.5

        logging.info(
            f"Initialized {name} with lengths - COXA: {COXA}cm, FEMUR: {FEMUR}cm, TIBIA: {TIBIA}cm"
        )
        logging.info(
            f"Reach limits - MAX: {self.MAX_REACH:.1f}cm, MIN: {self.MIN_REACH:.1f}cm"
        )

        self.theta1 = 0.0  # coxa angle
        self.theta2 = 0.0  # femur angle
        self.theta3 = 0.0  # tibia angle

        self.parts = parts or {}
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.joints = [np.array([0.0, 0.0, 0.0]) for _ in range(4)]

        # DH parameters [θ, d, a, α]
        # Coxa: Rotation in horizontal plane (Z axis)
        # Femur: Rotation in vertical plane (Y axis)
        # Tibia: Rotation in vertical plane (Y axis)
        self.leg_model = rtb.DHRobot(
            [
                rtb.RevoluteDH(
                    d=0,  # Z offset
                    a=0,  # X offset
                    alpha=0,  # Rotation around X axis
                ),
                rtb.RevoluteDH(
                    d=0,  # Z offset
                    a=self.COXA,  # X offset (coxa length)
                    alpha=pi / 2,  # 90 degree rotation around X axis
                ),
                rtb.RevoluteDH(
                    d=0,  # Z offset
                    a=self.FEMUR,  # X offset (femur length)
                    alpha=0,  # No rotation around X axis
                ),
            ],
            name=f"Leg_{name}",
        )

        try:
            self.update_joints()
        except Exception as e:
            logging.error(f"Error initializing {name}: {e}")
            raise

    def update_joints(self):
        """Update joint positions using forward kinematics."""
        try:
            angles = [self.theta1, self.theta2, self.theta3]

            # Initialize transforms list with origin
            transforms = [np.array([0.0, 0.0, 0.0])]

            # Calculate cumulative transforms
            T = np.eye(4)  # Start with identity matrix

            # First joint (Coxa) - Rotation in horizontal plane
            T1 = self.leg_model.links[0].A(angles[0])
            T = T @ np.array(T1)  # Convert SE3 to numpy array
            pos = np.array([T[0, 3], T[1, 3], T[2, 3]])
            transforms.append(pos)

            # Second joint (Femur) - Rotation in vertical plane
            T2 = self.leg_model.links[1].A(angles[1])
            T = T @ np.array(T2)  # Convert SE3 to numpy array
            pos = np.array([T[0, 3], T[1, 3], T[2, 3]])
            transforms.append(pos)

            # Third joint (Tibia) - Rotation in vertical plane
            T3 = self.leg_model.links[2].A(angles[2])
            T = T @ np.array(T3)  # Convert SE3 to numpy array
            pos = np.array([T[0, 3], T[1, 3], T[2, 3]])
            transforms.append(pos)

            # Add end effector position with TIBIA length
            # Tibia's direction is determined by the sum of femur and tibia angles
            total_angle = angles[1] + angles[2]  # Only femur and tibia angles
            final_pos = pos + np.array(
                [
                    self.TIBIA * cos(total_angle),  # X direction extension
                    0,  # Y remains constant
                    self.TIBIA * sin(total_angle),  # Z direction extension
                ]
            )
            transforms.append(final_pos)

            self.joints = transforms
            self.current_position = final_pos

        except Exception as e:
            logging.error(f"Error in update_joints: {e}")

    def set_angles(self, angles):
        """Set joint angles in degrees."""
        self.theta1, self.theta2, self.theta3 = map(radians, angles)
        self.update_joints()

    def get_angles(self):
        """Get joint angles in degrees."""
        return [degrees(self.theta1), degrees(self.theta2), degrees(self.theta3)]

    def forward_kinematics(self):
        """Perform forward kinematics to calculate joint positions."""
        theta1, theta2, theta3 = self.theta1, self.theta2, self.theta3

        try:
            # Use active leg's mounting point
            mount_pos = np.array(self.mount_positions[self.active_leg])

            # Adjust coxa angle (0 degrees = leg back, positive = counterclockwise)
            theta1 = -theta1 + pi / 4  # First invert angle, then add 45 degree offset

            # Coxa joint - rotation in horizontal plane
            coxa_end = np.array(
                [
                    mount_pos[0] + self.COXA * cos(theta1),
                    mount_pos[1] + self.COXA * sin(theta1),
                    0,
                ]
            )

            # Femur joint - rotation in vertical plane
            femur_proj = self.FEMUR * cos(theta2)  # Horizontal projection
            femur_height = self.FEMUR * sin(theta2)  # Vertical height

            femur_end = np.array(
                [
                    coxa_end[0] + femur_proj * cos(theta1),
                    coxa_end[1] + femur_proj * sin(theta1),
                    femur_height,
                ]
            )

            # Tibia joint - rotation in vertical plane
            total_angle = theta2 - theta3  # Femur angle minus tibia angle
            tibia_proj = self.TIBIA * cos(total_angle)  # Horizontal projection
            tibia_height = self.TIBIA * sin(total_angle)  # Vertical height

            tibia_end = np.array(
                [
                    femur_end[0] + tibia_proj * cos(theta1),
                    femur_end[1] + tibia_proj * sin(theta1),
                    femur_end[2] + tibia_height,
                ]
            )

            # Joint positions
            joints = [
                mount_pos,  # mount point (coxa start)
                coxa_end,  # coxa-femur joint
                femur_end,  # femur-tibia joint
                tibia_end,  # tip of the leg
            ]

            # Reachability check
            end_pos = joints[-1]
            if not self.is_position_reachable(end_pos):
                logging.warning(f"FK result position {end_pos} is not reachable!")

            # Save calculated position
            self.current_position = joints[-1]
            self.joints = joints
            return joints

        except Exception as e:
            logging.error(f"Error in forward kinematics: {e}")
            return None

    def inverse_kinematics(self, target=None):
        """Calculate inverse kinematics."""
        if target is None:
            return self.get_angles()

        target = np.array(target, dtype=np.float64)

        try:
            if not self.is_position_reachable(target):
                return self.get_angles()

            # Adjust target position relative to mount point
            mount_pos = np.array(self.mount_positions.get(self.active_leg, [0, 0, 0]))
            local_target = target - mount_pos

            # Calculate coxa angle in horizontal plane
            theta1 = (
                -atan2(local_target[1], local_target[0]) + pi / 4
            )  # Offset and direction adjustment

            # Subtract COXA length
            coxa_end = np.array(
                [
                    mount_pos[0] + self.COXA * cos(theta1),
                    mount_pos[1] + self.COXA * sin(theta1),
                    0,
                ]
            )

            femur_target = target - coxa_end

            # Femur-Tibia plane distance calculation
            L = np.linalg.norm(femur_target)

            # Femur and tibia angles calculation
            cos_theta3 = (L**2 - self.FEMUR**2 - self.TIBIA**2) / (
                2 * self.FEMUR * self.TIBIA
            )

            if abs(cos_theta3) > 1:
                return self.get_angles()

            theta3 = acos(cos_theta3)  # Tibia angle

            # Femur angle
            theta2 = atan2(
                femur_target[2], sqrt(femur_target[0] ** 2 + femur_target[1] ** 2)
            ) + atan2(self.TIBIA * sin(theta3), self.FEMUR + self.TIBIA * cos(theta3))

            # Convert angles to degrees
            angles = [degrees(theta1), degrees(theta2), degrees(theta3)]

            # Solution validation
            temp_angles = self.get_angles()
            self.set_angles(angles)
            end_pos = self.current_position
            error = np.linalg.norm(end_pos - target)

            if error > 0.1:
                self.set_angles(temp_angles)
                return temp_angles

            return angles

        except Exception as e:
            return self.get_angles()

    def is_position_reachable(self, target):
        """Check if target position is reachable."""
        target = np.array(target)
        mount_pos = np.array(self.mount_positions.get(self.active_leg, [0, 0, 0]))
        local_target = target - mount_pos  # Relative position relative to mount point

        # Check horizontal and vertical distances separately
        horizontal_distance = np.linalg.norm(local_target[:2])  # Only X-Y plane
        vertical_distance = abs(local_target[2])  # Z axis
        total_distance = np.linalg.norm(local_target)  # Total distance

        # Total length required - home position
        total_length = self.COXA + self.FEMUR + self.TIBIA

        # Maximum horizontal extension - full length allowed
        max_horizontal = total_length
        # Maximum vertical extension - 95% of FEMUR + TIBIA
        max_vertical = (self.FEMUR + self.TIBIA) * 0.95
        # Minimum horizontal distance - half COXA length
        min_horizontal = self.COXA * 0.5

        # Tolerance
        tolerance = 0.5  # 0.5cm tolerance

        # Home position check - full length leg must be valid
        if (
            abs(horizontal_distance - total_length) < tolerance
            and vertical_distance < tolerance
        ):
            return True

        if horizontal_distance > (max_horizontal + tolerance):
            return False

        if vertical_distance > (max_vertical + tolerance):
            return False

        if horizontal_distance < (min_horizontal - tolerance):
            return False

        return True

    def get_current_position(self):
        """Get current end effector position."""
        return self.current_position

    def get_joint_positions(self):
        """Get all joint positions."""
        return self.joints

    def current_coordinates(self):
        """Get current end effector coordinates."""
        return self.current_position

    def plot(self, ax):
        """Plot the leg in 3D."""
        joint_locations = self.forward_kinematics()
        if joint_locations is None:
            logging.error("Cannot plot: No valid joint positions")
            return

        # Convert list of lists to numpy array
        joint_locations = np.array(joint_locations)

        joint_names = ["Coxa", "Femur", "Tibia", "End"]
        colors = ["red", "green", "blue", "purple"]

        # Plot each joint and connection
        for idx, point in enumerate(joint_locations):
            # Plot joint point
            ax.scatter(point[0], point[1], point[2], color=colors[idx], s=100)

            # Add joint name
            ax.text(
                point[0],
                point[1],
                point[2] + 1,
                f"{joint_names[idx]}",
                horizontalalignment="center",
                verticalalignment="bottom",
                color=colors[idx],
                fontweight="bold",
            )

            # Plot connection to previous joint
            if idx > 0:
                prev_point = joint_locations[idx - 1]
                # Draw straight line for all joints
                ax.plot(
                    [prev_point[0], point[0]],
                    [prev_point[1], point[1]],
                    [prev_point[2], point[2]],
                    color=colors[idx],
                    linewidth=2,
                )

                # Add angle label
                ax.text(
                    prev_point[0],
                    prev_point[1],
                    prev_point[2] - 1,
                    f"θ{idx}={self.get_angles()[idx-1]:.1f}°",
                    horizontalalignment="right",
                    verticalalignment="top",
                    color=colors[idx],
                )

        # Add end effector coordinates
        end_pos = joint_locations[-1]
        ax.text(
            end_pos[0],
            end_pos[1],
            end_pos[2] - 2,
            f"End:\n({end_pos[0]:.1f},\n{end_pos[1]:.1f},\n{end_pos[2]:.1f})",
            horizontalalignment="left",
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Plot body outline
        self.plot_body(ax)

        # Set axis labels
        ax.set_xlabel("X (cm)")
        ax.set_ylabel("Y (cm)")
        ax.set_zlabel("Z (cm)")

        # Equal aspect ratio
        max_range = (
            np.array(
                [
                    joint_locations[:, 0].max() - joint_locations[:, 0].min(),
                    joint_locations[:, 1].max() - joint_locations[:, 1].min(),
                    joint_locations[:, 2].max() - joint_locations[:, 2].min(),
                ]
            ).max()
            / 2.0
        )

        mid_x = (joint_locations[:, 0].max() + joint_locations[:, 0].min()) * 0.5
        mid_y = (joint_locations[:, 1].max() + joint_locations[:, 1].min()) * 0.5
        mid_z = (joint_locations[:, 2].max() + joint_locations[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.grid(True)
        ax.view_init(elev=30, azim=225)

    def plot_body(self, ax):
        """Plot robot body outline."""
        # Body corner points
        corners = np.array(
            [
                [self.BODY_LENGTH / 2, self.BODY_WIDTH / 2, 0],  # Right front
                [-self.BODY_LENGTH / 2, self.BODY_WIDTH / 2, 0],  # Left front
                [-self.BODY_LENGTH / 2, -self.BODY_WIDTH / 2, 0],  # Left rear
                [self.BODY_LENGTH / 2, -self.BODY_WIDTH / 2, 0],  # Right rear
            ]
        )

        # Body lines
        for i in range(4):
            j = (i + 1) % 4
            ax.plot(
                [corners[i, 0], corners[j, 0]],
                [corners[i, 1], corners[j, 1]],
                [corners[i, 2], corners[j, 2]],
                "k--",
                linewidth=1,
                alpha=0.5,
            )

        # Leg mounting points
        for name, pos in self.mount_positions.items():
            color = "red" if name == self.active_leg else "black"
            ax.scatter(pos[0], pos[1], pos[2], color=color, s=50)
            ax.text(
                pos[0],
                pos[1],
                pos[2] + 0.5,
                name,
                horizontalalignment="center",
                verticalalignment="bottom",
                color=color,
                fontsize=8,
                fontweight="bold" if name == self.active_leg else "normal",
            )

    def set_active_leg(self, leg_name):
        """Set which leg to control."""
        if leg_name in self.mount_positions:
            self.active_leg = leg_name
            self.update_joints()
        else:
            logging.error(f"Invalid leg name: {leg_name}")
