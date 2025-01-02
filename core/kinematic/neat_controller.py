import math
import sys
import neat
import numpy as np

""" The NEAT controller class. Responsible for managing the ANN queries at each time step
"""

stationary = [0.18, 0, 0, 0, 0] * 6


class Controller:
    def __init__(
        self,
        params,
        body_height=0.1,  # 10 cm yükseklik
        velocity=0.25,  # 25 cm/s hedef hız
        period=0.4,  # 0.5 saniye periyot (daha hızlı adımlar)
        crab_angle=0.0,  # Düz ileri
        ann=None,
    ):
        self.params = params
        self.body_height = body_height
        self.velocity = velocity
        self.period = period
        self.crab_angle = crab_angle
        self.ann = ann

        # Zaman ve array boyutları
        self.dt = 1 / 240  # Zaman adımı
        self.array_dim = int(np.around(period / self.dt))  # Array boyutu
        self.count = 0  # Bacak sayacı

        # Başlangıç açıları (radyan)
        self.default_angles = {
            # Sağ bacaklar (1, 5, 6)
            "right": {
                "coxa": 0.0,  # Yatay düzlemde 0 derece
                "femur": np.pi / 6,  # 30 derece yukarı
                "tibia": np.pi / 2,  # 90 derece aşağı (ters yön)
            },
            # Sol bacaklar (2, 3, 4)
            "left": {
                "coxa": 0.0,  # Yatay düzlemde 0 derece
                "femur": np.pi / 6,  # 30 derece yukarı
                "tibia": -np.pi / 2,  # -90 derece aşağı
            },
        }

        # IMU verileri
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        self.height = body_height  # cm
        self.period = 0.4  # Periyodu 0.4 saniyeye düşürüyorum

    def joint_angles(self, t):
        """Eklem açılarını hesapla"""
        try:
            # Zaman indeksini hesapla
            k = int(((t % self.period) / self.period) * self.array_dim)

            # Her bacak için yörünge oluştur
            positions = np.zeros((18, self.array_dim))
            velocities = np.zeros((18, self.array_dim))
            angles = np.zeros((18, self.array_dim))
            speeds = np.zeros((18, self.array_dim))

            # Her bacak için yeni yörüngeler oluştur
            params = np.array(self.params).reshape(6, 5)
            for leg_index in range(6):
                foot_positions, foot_velocities = self.__leg_traj(
                    leg_index, params[leg_index]
                )
                joint_angles, joint_speeds = self.__inverse_kinematics(
                    foot_positions, foot_velocities
                )

                # Her bacak için 3 eklem açısı var
                start_idx = leg_index * 3
                angles[start_idx : start_idx + 3, :] = joint_angles
                speeds[start_idx : start_idx + 3, :] = joint_speeds
                positions[start_idx : start_idx + 3, :] = foot_positions
                velocities[start_idx : start_idx + 3, :] = foot_velocities

            # Tüm bacakların açılarını al
            all_angles = []
            for leg_idx in range(6):
                start_idx = leg_idx * 3
                leg_angles = angles[start_idx : start_idx + 3, k]
                all_angles.extend(leg_angles)

            return np.array(all_angles)

        except Exception as e:
            # Hata durumunda varsayılan açıları döndür
            default_angles = []
            for leg in range(6):
                is_right = leg in [0, 4, 5]  # 1, 5, 6 numaralı bacaklar
                side = "right" if is_right else "left"
                default_angles.extend(
                    [
                        self.default_angles[side]["coxa"],
                        self.default_angles[side]["femur"],
                        self.default_angles[side]["tibia"],
                    ]
                )
            return np.array(default_angles)

    def __leg_traj(self, leg_index, leg_params):
        """Her bir bacak için yörünge oluştur"""
        try:
            # Dikdörtgen konfigürasyonu için bacak açıları
            # Her bacağın yere göre hareket yönünü belirler
            leg_angles = {
                0: -np.pi
                / 4,  # Leg 1: Sağ ön köşe (-45 derece) - Sola ve öne doğru hareket
                1: np.pi
                / 4,  # Leg 2: Sol ön köşe (45 derece) - Sağa ve öne doğru hareket
                2: np.pi / 2,  # Leg 3: Sol orta (90 derece) - Direkt sağa doğru hareket
                3: 3
                * np.pi
                / 4,  # Leg 4: Sol arka köşe (135 derece) - Sağa ve arkaya doğru hareket
                4: -3
                * np.pi
                / 4,  # Leg 5: Sağ arka köşe (-135 derece) - Sola ve arkaya doğru hareket
                5: -np.pi
                / 2,  # Leg 6: Sağ orta (-90 derece) - Direkt sola doğru hareket
            }
            leg_angle = float(leg_angles[leg_index])  # Bacağın hareket açısı

            # NEAT parametreleri - her bacak için özelleştirilmiş hareket parametreleri
            radius = float(
                leg_params[0]
            )  # Bacağın merkeze olan uzaklığı (coxa hareketi için)
            offset = float(leg_params[1])  # Bacağın yatay düzlemdeki açısal ofseti
            step_height = float(
                leg_params[2]
            )  # Adım yüksekliği (havadaki maksimum yükseklik)
            phase = float(leg_params[3])  # Faz kayması (bacaklar arası zamanlama)
            duty_factor = float(
                leg_params[4]
            )  # Görev faktörü (yerde kalma süresi oranı)

            # İlerleme mesafesini hesapla
            stride = float(
                self.velocity * duty_factor * self.period
            )  # Bir adımda alınan mesafe
            # velocity: hedef hız (m/s)
            # duty_factor: yerde kalma süresi (0-1 arası)
            # period: bir adım döngüsünün süresi (saniye)

            # Bacak yüksekliklerini ayarla
            base_height = float(
                -self.body_height
            )  # Yerdeki pozisyon (negatif çünkü aşağı yön)
            mid_height = float(base_height + step_height)  # Havadaki maksimum yükseklik

            # Orta nokta (mid): Bacağın merkez pozisyonu
            # Bu nokta etrafında bacak hareket eder
            # Daha dengeli hareket için offset ve radius çarpanlarını azalt

            mid = np.array(
                [
                    float(
                        radius * 1.4 * np.cos(offset)
                    ),  # X koordinatı (daha küçük radius)
                    float(
                        radius * 1.5 * np.sin(offset)
                    ),  # Y koordinatı (daha küçük radius)
                    float(mid_height),  # Z koordinatı
                ]
            )

            # Başlangıç noktası (start): Adımın başladığı yer
            # Bacak açısına göre stride kadar ileri pozisyon
            start = np.array(
                [
                    float(mid[0] + stride * np.cos(-leg_angle + self.crab_angle)),  # X
                    float(mid[1] + stride * np.sin(-leg_angle + self.crab_angle)),  # Y
                    float(base_height),  # Z (yerde)
                ]
            )

            # Bitiş noktası (end): Adımın bittiği yer
            # Bacak açısına göre stride kadar geri pozisyon
            end = np.array(
                [
                    float(mid[0] - stride * np.cos(-leg_angle + self.crab_angle)),  # X
                    float(mid[1] - stride * np.sin(-leg_angle + self.crab_angle)),  # Y
                    float(base_height),  # Z (yerde)
                ]
            )

            # Destek fazı hesaplamaları (bacak yerde iken)
            support_dim = int(
                np.around(self.array_dim * duty_factor)
            )  # Yerde kalma süresi
            support_positions = np.zeros((3, support_dim))  # Yerdeki pozisyonlar
            support_velocities = np.zeros((3, support_dim))  # Yerdeki hızlar

            # Destek fazı için lineer interpolasyon
            # Start'tan end'e doğrusal hareket
            for i in range(3):  # x, y, z koordinatları için
                support_positions[i, :] = np.linspace(start[i], end[i], support_dim)
                support_velocities[i, :] = (
                    np.ones(support_dim) * (end[i] - start[i]) / (support_dim * self.dt)
                )

            # Salınım fazı hesaplamaları (bacak havada iken)
            swing_dim = self.array_dim - support_dim  # Havada kalma süresi
            swing_positions = np.zeros((3, swing_dim))  # Havadaki pozisyonlar
            swing_velocities = np.zeros((3, swing_dim))  # Havadaki hızlar

            # Salınım fazı için parabolik yörünge
            # End'den start'a parabolik hareket
            t = np.linspace(0, 1, swing_dim)  # Zaman parametresi
            for i in range(3):
                if (
                    i == 2
                ):  # z koordinatı için parabolik hareket (yerden yükselip alçalma)
                    swing_positions[i, :] = (
                        end[i] + (start[i] - end[i]) * t + 4 * step_height * t * (1 - t)
                    )
                    swing_velocities[i, :] = (
                        (start[i] - end[i]) + 4 * step_height * (1 - 2 * t)
                    ) / (swing_dim * self.dt)
                else:  # x ve y koordinatları için lineer hareket
                    swing_positions[i, :] = end[i] + (start[i] - end[i]) * t
                    swing_velocities[i, :] = (
                        np.ones(swing_dim) * (start[i] - end[i]) / (swing_dim * self.dt)
                    )

            # Tüm fazları birleştir
            positions = np.hstack(
                [support_positions, swing_positions]
            )  # Tüm pozisyonlar
            velocities = np.hstack([support_velocities, swing_velocities])  # Tüm hızlar

            # Faz kaydırması uygula (bacaklar arası zamanlama)
            phase_shift = int(np.around(phase * self.array_dim))  # Faz kayması miktarı
            positions = np.roll(positions, phase_shift, axis=1)  # Pozisyonları kaydır
            velocities = np.roll(velocities, phase_shift, axis=1)  # Hızları kaydır

            return positions, velocities

        except Exception as e:
            return np.zeros((3, self.array_dim)), np.zeros((3, self.array_dim))

    def __inverse_kinematics(self, foot_position, foot_speed):
        """Ters kinematik hesaplama"""
        try:
            l_1, l_2, l_3 = 0.0435, 0.080, 0.155  # Bacak uzunlukları

            x, y, z = foot_position
            dx, dy, dz = foot_speed

            # Coxa açısı hesaplama
            theta_1 = -np.arctan2(y, x)  # Tüm bacaklar için -1 ile çarp

            c_1, s_1 = np.cos(theta_1), np.sin(theta_1)
            c_3 = (
                (x - l_1 * c_1) ** 2 + (y - l_1 * s_1) ** 2 + z**2 - l_2**2 - l_3**2
            ) / (2 * l_2 * l_3)
            s_3 = -np.sqrt(np.maximum(1 - c_3**2, 0))  # maximum ensures not negative

            theta_2 = np.arctan2(
                z, (np.sqrt((x - l_1 * c_1) ** 2 + (y - l_1 * s_1) ** 2))
            ) - np.arctan2((l_3 * s_3), (l_2 + l_3 * c_3))
            theta_3 = np.arctan2(s_3, c_3)

            # Femur açıları
            leg_index = self.count % 6  # Get current leg index (0-5)
            if leg_index in [0, 4, 5]:  # Legs 1, 5, 6 (zero-based indexing)
                theta_3 = -theta_3

            # Açı limitlerini uygula
            theta_1 = np.clip(theta_1, -np.pi / 2, np.pi / 2)
            theta_2 = np.clip(theta_2, -np.pi / 2, np.pi / 2)
            theta_3 = np.clip(theta_3, -np.pi / 2, np.pi / 2)

            c_2, s_2 = np.cos(theta_2), np.sin(theta_2)
            c_23 = np.cos(theta_2 + theta_3)

            with np.errstate(all="ignore"):
                theta_dot_1 = (dy * c_1 - dx * s_1) / (l_1 + l_3 * c_23 + l_2 * c_2)
                theta_dot_2 = (1 / l_2) * (
                    dz * c_2
                    - dx * c_1 * s_2
                    - dy * s_1 * s_2
                    + (c_3 / s_3) * (dz * s_2 + dx * c_1 * c_2 + dy * c_2 * s_1)
                )
                theta_dot_3 = -(1 / l_2) * (
                    dz * c_2
                    - dx * c_1 * s_2
                    - dy * s_1 * s_2
                    + ((l_2 + l_3 * c_3) / (l_3 * s_3))
                    * (dz * s_2 + dx * c_1 * c_2 + dy * c_2 * s_1)
                )

            # Hız değerlerini temizle
            theta_dot_1 = np.nan_to_num(theta_dot_1, nan=0.0, posinf=0.0, neginf=0.0)
            theta_dot_2 = np.nan_to_num(theta_dot_2, nan=0.0, posinf=0.0, neginf=0.0)
            theta_dot_3 = np.nan_to_num(theta_dot_3, nan=0.0, posinf=0.0, neginf=0.0)

            joint_angles = np.array([theta_1, theta_2, theta_3])
            joint_speeds = np.array([theta_dot_1, theta_dot_2, theta_dot_3])

            self.count += 1
            return joint_angles, joint_speeds

        except Exception as e:
            return np.zeros(3), np.zeros(3)

    def IMU_feedback(self, orientation):
        """IMU geri bildirimi al"""
        self.roll, self.pitch, self.yaw = orientation


# reshapes a 32 length array of floats range 0.0 - 1.0 into the range expected by the controller
def reshape(x):
    x = np.array(x)
    # get body height and velocity
    height = x[0] * 0.2
    velocity = x[1] * 0.5
    leg_params = x[2:].reshape((6, 5))
    # radius, offset, step_height, phase, duty_cycle
    param_min = np.array([0.0, -1.745, 0.01, 0.0, 0.0])
    param_max = np.array([0.3, 1.745, 0.2, 1.0, 1.0])
    # scale and shifted params into the ranges expected by controller
    leg_params = leg_params * (param_max - param_min) + param_min

    return height, velocity, leg_params


if __name__ == "__main__":
    import time

    from core.kinematic.gait_config import tripod_gait

    start = time.perf_counter()
    ctrl = Controller(tripod_gait)
    end = time.perf_counter()

    print((end - start) * 1000)


# # 10x10x10 grid, çalıştırma no 1, 100 nesil (varsayılan)
# python neat_map_elites.py 10 1

# # 20x20x20 grid, çalıştırma no 2, 1000 nesil
# python neat_map_elites.py 20 2 --generations 1000


# python neat_map_elites.py 10 1 --visualize --generations 20
