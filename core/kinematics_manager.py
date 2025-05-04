import numpy as np
from loguru import logger

class KinematicsManager:
    """Main class that manages kinematic calculations"""
    def __init__(self):
        
        # Debug mode
        self.debug = True
        
        # Robot dimensions (meters) - taken from URDF
        self.BODY_LENGTH = 0.18  # 172mm -> 0.172m
        self.BODY_WIDTH = 0.12   # 110mm -> 0.110m
        
        # Leg lengths (meters)
        self.COXA = 0.043   # 43.5mm -> 0.0435m
        self.FEMUR = 0.08    # 80mm -> 0.08m
        self.TIBIA = 0.165
        
        # Safety parameters
        self.MIN_REACH = 0.07
        self.MAX_REACH = 0.285  # Adjusted from 0.28 to 0.285
        self.MAX_REACH_FACTOR = 0.95  # 95% of maximum reach
        self.MIN_REACH_FACTOR = 0.95  # 95% of minimum reach
        self.POSITION_TOLERANCE = 0.005  # 5mm position tolerance
        
        # URDF axis limits
        self.JOINT_LIMITS = {
            'coxa': (-90, 90),   # degrees
            'femur': (-90, 90),
            'tibia': (-90, 90)
        }
        
        # Servo connection angles (offset angles)
        self.SERVO_OFFSETS = {
            'coxa': -8.0,
            'femur': 35.0,
            'tibia': 68.0
        }
        
        # Leg connection points (mount positions)
        self.mount_positions = np.zeros((6, 3))
        self.init_mount_positions()
        
        self.current_position = None
        self.joints = None
        self.active_leg = 1

    def init_mount_positions(self):
        """Initialize leg connection points"""
        # Calculate leg connection points based on body dimensions
        half_length = self.BODY_LENGTH / 2
        half_width = self.BODY_WIDTH / 2
        z_mount = 0.0  # Z coordinate of connection points
        
        # Leg order: 1-Right front, 2-Left front, 3-Left middle, 4-Left rear, 5-Right rear, 6-Right middle
        
        # Right front leg (1)
        self.mount_positions[0, 0] = half_length
        self.mount_positions[0, 1] = -half_width
        self.mount_positions[0, 2] = z_mount
        
        # Left front leg (2)
        self.mount_positions[1, 0] = half_length
        self.mount_positions[1, 1] = half_width
        self.mount_positions[1, 2] = z_mount
        
        # Left middle leg (3)
        self.mount_positions[2, 0] = 0
        self.mount_positions[2, 1] = half_width
        self.mount_positions[2, 2] = z_mount
        
        # Left rear leg (4)
        self.mount_positions[3, 0] = -half_length
        self.mount_positions[3, 1] = half_width
        self.mount_positions[3, 2] = z_mount
        
        # Right rear leg (5)
        self.mount_positions[4, 0] = -half_length
        self.mount_positions[4, 1] = -half_width
        self.mount_positions[4, 2] = z_mount
        
        # Right middle leg (6)
        self.mount_positions[5, 0] = 0
        self.mount_positions[5, 1] = -half_width
        self.mount_positions[5, 2] = z_mount

    def calculate_inverse(self, foot_position, foot_speed, leg_num=None):
        """
        Calculate inverse kinematics - calculate joint angles from foot position
        
        Args:
            foot_position: Foot position [x, y, z]
            foot_speed: Foot speed [dx, dy, dz]
            leg_num: Leg number (1-6)
            
        Returns:
            tuple: (joint_angles, joint_speeds)
        """
        try:
            l_1, l_2, l_3 = self.COXA, self.FEMUR, self.TIBIA
            x, y, z = foot_position
            dx, dy, dz = foot_speed

            # Coxa angle (rotation around z-axis)
            theta_1 = -np.arctan2(y, x)

            # Trigonometric calculations
            c_1, s_1 = np.cos(theta_1), np.sin(theta_1)
            
            # Distance from coxa to foot
            coxa_to_foot_xy = np.sqrt((x - l_1 * c_1)**2 + (y - l_1 * s_1)**2)
            coxa_to_foot = np.sqrt(coxa_to_foot_xy**2 + z**2)
            
            
            if coxa_to_foot > self.MAX_REACH:
                logger.warning(f"Position out of reach: {coxa_to_foot:.3f}m > {self.MAX_REACH:.3f}m")
                # Adjust to reach limit
                scale_factor = (self.MAX_REACH) / coxa_to_foot
                coxa_to_foot_xy *= scale_factor
                z *= scale_factor
                coxa_to_foot = self.MAX_REACH
            
            if coxa_to_foot < self.MIN_REACH:
                logger.warning(f"Position too close: {coxa_to_foot:.3f}m < {self.MIN_REACH:.3f}m")
                # Adjust to minimum reach limit
                scale_factor = (self.MIN_REACH) / coxa_to_foot
                coxa_to_foot_xy *= scale_factor
                z *= scale_factor
                coxa_to_foot = self.MIN_REACH
            
            # Calculate tibia angle using cosine theorem
            phi1_val = (l_2**2 + coxa_to_foot**2 - l_3**2) / (2 * l_2 * coxa_to_foot)
            phi1_val = np.clip(phi1_val, -1.0, 1.0)
            phi1 = np.arccos(phi1_val)
            
            # Femur angle
            elevation_angle = np.arctan2(z, coxa_to_foot_xy)
            inner_angle = np.arctan2(l_3 * np.sin(phi1), l_2 + l_3 * np.cos(phi1))
            theta_2 = elevation_angle - inner_angle
            
            # Apply angle transformations based on leg side
            if leg_num is not None:
                if leg_num in [1, 5, 6]:  # Right legs
                    phi1 = -phi1  # Invert tibia angle
                else:  # Left legs
                    theta_2 = -theta_2  # Invert femur angle

            # Joint angles
            joint_angles = np.array([theta_1, theta_2, phi1])
            
            # Joint velocities (using Jacobian)
            c_2, s_2 = np.cos(theta_2), np.sin(theta_2)
            c_23 = np.cos(theta_2 + phi1)
            s_23 = np.sin(theta_2 + phi1)
            
            # Calculate Jacobian matrix
            J = np.array([
                [-s_1*(l_1 + l_2*c_2 + l_3*c_23), -c_1*(l_2*s_2 + l_3*s_23), -l_3*c_1*s_23],
                [c_1*(l_1 + l_2*c_2 + l_3*c_23), -s_1*(l_2*s_2 + l_3*s_23), -l_3*s_1*s_23],
                [0, l_2*c_2 + l_3*c_23, l_3*c_23]
            ])
            
            # Calculate inverse of Jacobian matrix
            try:
                J_inv = np.linalg.pinv(J)
                joint_speeds = J_inv.dot(np.array([dx, dy, dz]))
            except np.linalg.LinAlgError:
                # Zero velocities in case of singular matrix
                joint_speeds = np.zeros(3)
                if self.debug:
                    logger.warning("Singular Jacobian matrix encountered")
            
            # Clean NaN and infinite values
            joint_speeds = np.nan_to_num(joint_speeds, nan=0.0, posinf=0.0, neginf=0.0)
            
            return joint_angles, joint_speeds
            
        except Exception as e:
            logger.error(f"Inverse kinematics error: {e}")
            return np.zeros(3), np.zeros(3)

    def forward_kinematics(self, leg_num, joint_angles):
        """
        Calculate forward kinematics - calculate foot position from joint angles
        
        Args:
            leg_num: Leg number (0-5)
            joint_angles: Joint angles [theta_1, theta_2, theta_3]
            
        Returns:
            numpy.ndarray: Foot position [x, y, z]
        """
        try:
            l_1, l_2, l_3 = self.COXA, self.FEMUR, self.TIBIA
            theta_1, theta_2, theta_3 = joint_angles
            
            # Get mount position
            mount_x = self.mount_positions[leg_num, 0]
            mount_y = self.mount_positions[leg_num, 1]
            mount_z = self.mount_positions[leg_num, 2]
            
            # Coxa end point
            Xa = l_1 * np.cos(theta_1)
            Ya = l_1 * np.sin(theta_1)
            Za = 0
            
            # Femur end point
            G2 = np.sin(theta_2) * l_2
            P1 = np.cos(theta_2) * l_2
            Xc = np.cos(theta_1) * P1
            Yc = np.sin(theta_1) * P1
            
            # Intermediate calculations for tibia end point
            femur_tibia_angle = theta_2 + theta_3
            Xb = np.cos(theta_1) * (P1 + l_3 * np.cos(femur_tibia_angle))
            Yb = np.sin(theta_1) * (P1 + l_3 * np.cos(femur_tibia_angle))
            G1 = G2 + l_3 * np.sin(femur_tibia_angle)
            
            # Calculate all joint positions
            joint_positions = np.array([
                [mount_x, mount_y, mount_z],              # Starting point
                [mount_x + Xa, mount_y + Ya, mount_z + Za],  # Coxa end
                [mount_x + Xa + Xc, mount_y + Ya + Yc, mount_z + G2],  # Femur end
                [mount_x + Xa + Xb, mount_y + Ya + Yb, mount_z + G1]   # Tibia end (foot)
            ])
            
            if self.debug:
                logger.debug(f"Leg {leg_num + 1} Forward Kinematics:")
                logger.debug(f"Mount Position: [{mount_x:.3f}, {mount_y:.3f}, {mount_z:.3f}]")
                logger.debug(f"Coxa End: [{Xa:.3f}, {Ya:.3f}, {Za:.3f}]")
                logger.debug(f"Femur End: [{Xc:.3f}, {Yc:.3f}, {G2:.3f}]")
                logger.debug(f"Tibia End: [{Xb:.3f}, {Yb:.3f}, {G1:.3f}]")
                logger.debug(f"Final Position: [{joint_positions[3,0]:.3f}, {joint_positions[3,1]:.3f}, {joint_positions[3,2]:.3f}]")
            
            return joint_positions[3]  # Only foot end position
            
        except Exception as e:
            logger.error(f"Forward kinematics error: {e}")
            return np.zeros(3)

    def calculate_velocities(self, joint_angles: np.ndarray, joint_velocities: np.ndarray) -> np.ndarray:
        """Calculate joint velocities"""
        try:
            theta_1, theta_2, theta_3 = joint_angles
            omega_1, omega_2, omega_3 = joint_velocities
            
            # Trigonometric calculations
            c1, s1 = np.cos(theta_1), np.sin(theta_1)
            c2, s2 = np.cos(theta_2), np.sin(theta_2)
            c23 = np.cos(theta_2 + theta_3)
            s23 = np.sin(theta_2 + theta_3)
            
            # Calculate Jacobian matrix - using COXA, FEMUR, TIBIA
            J = np.array([
                [-s1*(self.COXA + self.FEMUR*c2 + self.TIBIA*c23), -c1*(self.FEMUR*s2 + self.TIBIA*s23), -self.TIBIA*c1*s23],
                [c1*(self.COXA + self.FEMUR*c2 + self.TIBIA*c23),  -s1*(self.FEMUR*s2 + self.TIBIA*s23), -self.TIBIA*s1*s23],
                [0,                                         self.FEMUR*c2 + self.TIBIA*c23,        self.TIBIA*c23]
            ])
            
            # Calculate end point velocities
            velocities = J.dot(np.array([omega_1, omega_2, omega_3]))
            
            return velocities
            
        except Exception as e:
            logger.error(f"Velocity calculation error: {e}")
            return np.zeros(3)
            
    def degrees_to_radians(self, angles_deg):
        """Convert degrees to radians"""
        return np.radians(angles_deg)
        
    def radians_to_degrees(self, angles_rad):
        """Convert radians to degrees"""
        return np.degrees(angles_rad)
        
    def apply_servo_offsets(self, angles_deg):
        """Apply servo offset angles"""
        adjusted = np.zeros(3)
        names = ['coxa', 'femur', 'tibia']
        
        for i in range(3):
            adjusted[i] = angles_deg[i] - self.SERVO_OFFSETS[names[i]]
            
        return adjusted
        
    def remove_servo_offsets(self, angles_deg):
        """Remove servo offset angles"""
        adjusted = np.zeros(3)
        names = ['coxa', 'femur', 'tibia']
        
        for i in range(3):
            adjusted[i] = angles_deg[i] + self.SERVO_OFFSETS[names[i]]
            
        return adjusted
        
    def get_leg_position(self, leg_num, angles_deg):
        """
        Calculate foot position for a specific leg
        
        Args:
            leg_num: Leg number (0-5)
            angles_deg: Joint angles in degrees [coxa, femur, tibia]
            
        Returns:
            numpy.ndarray: Foot position [x, y, z]
        """
        # Convert degrees to radians
        angles_rad = self.degrees_to_radians(angles_deg)
        
        # Calculate forward kinematics
        position = self.forward_kinematics(leg_num, angles_rad)
        
        return position
        
    def get_leg_angles(self, leg_num, position, speed=np.zeros(3)):
        """
        Calculate joint angles for a specific leg
        
        Args:
            leg_num: Leg number (0-5)
            position: Foot position [x, y, z]
            speed: Foot speed [dx, dy, dz]
            
        Returns:
            tuple: (angles_deg, speeds_deg) - Joint angles and speeds in degrees
        """
        # Adjust according to mount position
        rel_position = position - self.mount_positions[leg_num]
        
        # Calculate inverse kinematics
        angles_rad, speeds_rad = self.calculate_inverse(rel_position, speed)
        
        # Convert radians to degrees
        angles_deg = self.radians_to_degrees(angles_rad)
        speeds_deg = self.radians_to_degrees(speeds_rad)
        
        return angles_deg, speeds_deg

    def plot_leg(self, leg_num, joint_angles=None):
        """
        Draw 3D visualization of a specific leg
        
        Args:
            leg_num: Leg number (0-5)
            joint_angles: Joint angles [theta_1, theta_2, theta_3]
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Calculate joint positions
            if joint_angles is None:
                joint_angles = np.zeros(3)
            joint_positions = self.forward_kinematics(leg_num, joint_angles)
            
            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Separate joint positions
            x = joint_positions[:, 0]
            y = joint_positions[:, 1]
            z = joint_positions[:, 2]
            
            # Draw leg segments
            ax.plot(x, y, z, '-o', linewidth=2, markersize=8)
            
            # Calculate segment midpoints
            coxa_mid = (joint_positions[0] + joint_positions[1]) / 2
            femur_mid = (joint_positions[1] + joint_positions[2]) / 2
            tibia_mid = (joint_positions[2] + joint_positions[3]) / 2
            
            # Add segment labels
            ax.text(coxa_mid[0], coxa_mid[1], coxa_mid[2], 'Coxa', fontsize=12, color='blue')
            ax.text(femur_mid[0], femur_mid[1], femur_mid[2], 'Femur', fontsize=12, color='blue')
            ax.text(tibia_mid[0], tibia_mid[1], tibia_mid[2], 'Tibia', fontsize=12, color='blue')
            
            # Set axes
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            # Add title
            ax.set_title(f'Leg {leg_num + 1} Visualization')
            
            # Set axis limits
            max_range = self.MAX_REACH
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            
            # Show plot
            plt.show()
            
        except Exception as e:
            logger.error(f"Plotting error: {e}")

    def plot_all_legs(self, joint_angles=None):
        """
        Draw 3D visualization of all legs
        
        Args:
            joint_angles: List of joint angles for each leg
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Create 3D plot
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # For each leg
            for leg_num in range(6):
                # Get joint angles
                if joint_angles is None:
                    angles = np.zeros(3)
                else:
                    angles = joint_angles[leg_num]
                
                # Calculate joint positions
                joint_positions = self.forward_kinematics(leg_num, angles)
                
                # Separate joint positions
                x = joint_positions[:, 0]
                y = joint_positions[:, 1]
                z = joint_positions[:, 2]
                
                # Draw leg segments
                ax.plot(x, y, z, '-o', linewidth=2, markersize=8, label=f'Leg {leg_num + 1}')
            
            # Set axes
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            
            # Add title
            ax.set_title('Hexapod Legs Visualization')
            
            # Set axis limits
            max_range = self.MAX_REACH * 1.5
            ax.set_xlim([-max_range, max_range])
            ax.set_ylim([-max_range, max_range])
            ax.set_zlim([-max_range, max_range])
            
            # Draw body
            body_x = [-self.BODY_LENGTH/2, self.BODY_LENGTH/2]
            body_y = [-self.BODY_WIDTH/2, self.BODY_WIDTH/2]
            body_z = [0, 0]
            ax.plot(body_x, [0, 0], [0, 0], 'k-', linewidth=3)
            ax.plot([0, 0], body_y, [0, 0], 'k-', linewidth=3)
            
            # Add body label
            ax.text(0, 0, 0, 'Body', fontsize=12, color='black')
            
            # Add legend
            ax.legend()
            
            # Show plot
            plt.show()
            
        except Exception as e:
            logger.error(f"Plotting error: {e}")

    def get_joint_positions(self, leg_num, joint_angles):
        """
        Return all joint positions (for visualization)
        """
        l_1, l_2, l_3 = self.COXA, self.FEMUR, self.TIBIA
        theta_1, theta_2, theta_3 = joint_angles
        mount_x = self.mount_positions[leg_num, 0]
        mount_y = self.mount_positions[leg_num, 1]
        mount_z = self.mount_positions[leg_num, 2]
        Xa = l_1 * np.cos(theta_1)
        Ya = l_1 * np.sin(theta_1)
        Za = 0
        G2 = np.sin(theta_2) * l_2
        P1 = np.cos(theta_2) * l_2
        Xc = np.cos(theta_1) * P1
        Yc = np.sin(theta_1) * P1
        femur_tibia_angle = theta_2 + theta_3
        Xb = np.cos(theta_1) * (P1 + l_3 * np.cos(femur_tibia_angle))
        Yb = np.sin(theta_1) * (P1 + l_3 * np.cos(femur_tibia_angle))
        G1 = G2 + l_3 * np.sin(femur_tibia_angle)
        joint_positions = np.array([
            [mount_x, mount_y, mount_z],
            [mount_x + Xa, mount_y + Ya, mount_z + Za],
            [mount_x + Xa + Xc, mount_y + Ya + Yc, mount_z + G2],
            [mount_x + Xa + Xb, mount_y + Ya + Yb, mount_z + G1]
        ])
        return joint_positions