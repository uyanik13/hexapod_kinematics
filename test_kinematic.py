import numpy as np
from core.kinematics_manager import KinematicsManager


class TestKinematic():
    def __init__(self):
        """Setup that runs at the beginning of the test"""
        self.km = KinematicsManager()

    def test_forward_inverse_consistency(self):
        """Test the consistency between forward and inverse kinematics"""
        # Test angles (degrees)
        test_angles = [
            # Normal stance
            ([0, 35, 65], "Normal stance"),
            ([0, 35, 90], "leg_down"),

            # # Turn positions
            # ([45, 35, 65], "Right turn"),
            # ([-45, -35, -65], "Left turn"),
            # # Extreme positions
            # ([0, 0, 0], "Fully extended"),
            # ([0, 45, 90], "Fully bent")
        ]
        
        for angles, description in test_angles:
            print(f"\nTest Case: {description}")
            print(f"Input angles: {angles}")
            
            for leg_index in range(6):
                print(f"\nLeg {leg_index+1}:")
                
                # Calculate forward kinematics
                position = self.km.forward_kinematics(leg_index, np.radians(angles))
                print(f"Forward Kinematics result  GIVEN_ANGLE: {angles} - Position RESULT: {position}")
                
                # Calculate inverse kinematics
                end_effector_pos = position
                result = self.km.calculate_inverse(end_effector_pos, [0, 0, 0], leg_index)
                
                # Check if inverse kinematics failed
                if result is None or result[0] is None:
                    print(f"Inverse Kinematics failed for leg {leg_index+1}")
                    continue
                    
                calculated_angles, _ = result
                joint_angles = np.degrees(calculated_angles)
                print(f"Inverse Kinematics result - GIVEN_POSITION: {end_effector_pos} - RESULT: {joint_angles}")
               
                
                # Calculate angle difference
                angle_diff = np.abs(angles - joint_angles)
                print(f"GIVEN_ANGLE: {angles}")
                print(f"CALCULATED_ANGLE: {joint_angles}")
                print(f"ANGLE_DIFF: {angle_diff}")
                
                # Position check
                recalc_position = self.km.forward_kinematics(leg_index, calculated_angles)
                position_diff = np.linalg.norm(position - recalc_position)
                print(f"Position difference: {position_diff} meters")

        # Set test angles (in radians)
        # test_angles = np.radians([0, 35, -65])  # 0°, 35°, 65°

        # # Calculate positions for one leg
        # positions = self.km.forward_kinematics(leg_num=0, joint_angles=test_angles)

        # Visualize results
        # self.km.plot_leg(leg_num=0, joint_angles=test_angles)
             


if __name__ == "__main__":
    test_kinematic = TestKinematic()
    test_kinematic.test_forward_inverse_consistency()
