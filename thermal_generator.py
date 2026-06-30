from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


class HotspotShape(Enum):
    CIRCULAR = "circular"
    IRREGULAR = "irregular"


@dataclass
class ThermalFrame:
    # Raw temperature matrix in Celsius.
    image: np.ndarray
    # Union hotspot mask for model training.
    mask: np.ndarray
    centers: list[Tuple[float, float]]
    temperatures: list[float]
    width: int
    height: int
    # FLIR-style rendered RGB image.
    thermal_image: Optional[np.ndarray] = None
    # Per-source binary masks: cpu, gpu, heatpipe, keyboard, fan, palm_rest.
    region_masks: dict[str, np.ndarray] = field(default_factory=dict)
    # Highest temperature pixel as global hotspot ground truth.
    hotspot_coordinate: Optional[Tuple[float, float]] = None
    hotspot_temperature: Optional[float] = None
    workload: str = "medium"


class ThermalImageGenerator:
    """Generate realistic laptop thermal scenes for synthetic data creation."""

    WORKLOAD_PROFILES = {
        "light": {
            "cpu_temp": (60.0, 72.0),
            "gpu_temp": (50.0, 62.0),
            "fan_temp": (42.0, 52.0),
            "keyboard_gain": (0.10, 0.16),
            "max_temp": (70.0, 78.0),
        },
        "medium": {
            "cpu_temp": (68.0, 82.0),
            "gpu_temp": (56.0, 72.0),
            "fan_temp": (46.0, 60.0),
            "keyboard_gain": (0.14, 0.22),
            "max_temp": (76.0, 86.0),
        },
        "heavy": {
            "cpu_temp": (78.0, 92.0),
            "gpu_temp": (66.0, 84.0),
            "fan_temp": (52.0, 68.0),
            "keyboard_gain": (0.20, 0.30),
            "max_temp": (84.0, 95.0),
        },
        "cpu_stress": {
            "cpu_temp": (84.0, 95.0),
            "gpu_temp": (50.0, 64.0),
            "fan_temp": (54.0, 72.0),
            "keyboard_gain": (0.18, 0.28),
            "max_temp": (88.0, 95.0),
        },
        "gpu_stress": {
            "cpu_temp": (64.0, 78.0),
            "gpu_temp": (76.0, 92.0),
            "fan_temp": (50.0, 70.0),
            "keyboard_gain": (0.17, 0.27),
            "max_temp": (84.0, 95.0),
        },
        "dual_stress": {
            "cpu_temp": (84.0, 95.0),
            "gpu_temp": (78.0, 92.0),
            "fan_temp": (58.0, 75.0),
            "keyboard_gain": (0.22, 0.34),
            "max_temp": (90.0, 95.0),
        },
    }

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        background_temp: float = 30.0,
        noise_std: float = 1.0,
        seed: Optional[int] = None,
    ):
        if (width, height) not in {(320, 240), (640, 480)}:
            raise ValueError("Supported resolutions are 320x240 and 640x480")
        self.width = width
        self.height = height
        self.background_temp = background_temp
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        hotspot_count: int = 1,
        hotspot_temp_range: Tuple[float, float] = (80.0, 100.0),
        hotspot_radius_range: Tuple[int, int] = (8, 20),
        shape: HotspotShape = HotspotShape.CIRCULAR,
        workload: Optional[str] = None,
    ) -> ThermalFrame:
        """Generate a realistic FLIR-like laptop thermal scene."""
        del hotspot_temp_range, hotspot_radius_range, shape

        scene_workload = workload or self._workload_from_hotspot_count(hotspot_count)
        if scene_workload not in self.WORKLOAD_PROFILES:
            raise ValueError(f"Unknown workload '{scene_workload}'")

        profile = self.WORKLOAD_PROFILES[scene_workload]
        ambient = float(np.clip(self.background_temp + self.rng.uniform(-2.5, 0.8), 22.0, 30.0))

        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        temp = np.full((self.height, self.width), ambient, dtype=np.float32)

        # Structural laptop gradients.
        top_bias = np.clip(1.0 - (yy / max(self.height - 1, 1)), 0.0, 1.0)
        center_bias = 1.0 - np.abs((xx - self.width * 0.5) / (self.width * 0.5 + 1e-6))
        temp += 1.8 * top_bias + 0.8 * center_bias

        cpu_cx = float(self.width * self.rng.uniform(0.34, 0.50))
        cpu_cy = float(self.height * self.rng.uniform(0.16, 0.25))
        gpu_cx = float(np.clip(cpu_cx + self.width * self.rng.uniform(0.10, 0.18), 0, self.width - 1))
        gpu_cy = float(np.clip(cpu_cy + self.height * self.rng.uniform(-0.03, 0.05), 0, self.height - 1))
        fan_cx = float(self.width * self.rng.uniform(0.72, 0.86))
        fan_cy = float(self.height * self.rng.uniform(0.14, 0.25))

        cpu_peak = float(self.rng.uniform(*profile["cpu_temp"]))
        gpu_peak = float(min(cpu_peak - self.rng.uniform(1.5, 7.0), self.rng.uniform(*profile["gpu_temp"])))
        fan_peak = float(self.rng.uniform(*profile["fan_temp"]))

        cpu_amp = max(cpu_peak - ambient, 1.0)
        gpu_amp = max(gpu_peak - ambient, 1.0)
        fan_amp = max(fan_peak - ambient, 0.5)

        cpu_blob = self._gaussian_2d(xx, yy, cpu_cx, cpu_cy, self.width * 0.055, self.height * 0.060)
        gpu_blob = self._gaussian_2d(xx, yy, gpu_cx, gpu_cy, self.width * 0.060, self.height * 0.064)
        fan_blob = self._gaussian_2d(xx, yy, fan_cx, fan_cy, self.width * 0.038, self.height * 0.050)

        # Heatpipe effect: elongated horizontal spread from CPU/GPU toward fan.
        pipe_cx = float((cpu_cx + fan_cx) * 0.5)
        pipe_cy = float((cpu_cy + fan_cy) * 0.5 + self.height * self.rng.uniform(-0.01, 0.01))
        heatpipe_blob = self._gaussian_2d(
            xx,
            yy,
            pipe_cx,
            pipe_cy,
            self.width * 0.17,
            self.height * 0.022,
        )

        temp += cpu_amp * cpu_blob
        temp += gpu_amp * gpu_blob
        temp += fan_amp * fan_blob
        temp += (0.28 * cpu_amp + 0.18 * gpu_amp) * heatpipe_blob

        # Keyboard conduction with smooth diffusion and gradients.
        source_energy = np.clip(cpu_blob + gpu_blob + 0.9 * heatpipe_blob, 0.0, 2.5).astype(np.float32)
        conduction = cv2.GaussianBlur(source_energy, (0, 0), sigmaX=self.width * 0.09, sigmaY=self.height * 0.07)
        conduction /= np.max(conduction) + 1e-6

        keyboard_mask = np.zeros((self.height, self.width), dtype=np.float32)
        key_x0 = int(self.width * 0.12)
        key_x1 = int(self.width * 0.88)
        key_y0 = int(self.height * 0.43)
        key_y1 = int(self.height * 0.75)
        keyboard_mask[key_y0:key_y1, key_x0:key_x1] = 1.0

        # Slight key-row pattern to mimic real laptop thermal texture.
        row_pattern = np.sin(np.linspace(0, np.pi * 10, self.height, dtype=np.float32))[:, None]
        row_pattern = 0.06 * (row_pattern + 1.0)
        keyboard_gain = float(self.rng.uniform(*profile["keyboard_gain"]))
        temp += keyboard_mask * (keyboard_gain * cpu_amp * conduction + row_pattern * (cpu_amp * 0.08))

        # Palm rest: generally cooler region.
        palm_y0 = int(self.height * 0.76)
        palm_cool = float(self.rng.uniform(1.5, 4.5))
        temp[palm_y0:, :] -= palm_cool

        # Fan ring texture around cooling region.
        fan_dist = np.sqrt((xx - fan_cx) ** 2 + (yy - fan_cy) ** 2)
        ring = np.exp(-((fan_dist - self.width * 0.03) ** 2) / (2.0 * (self.width * 0.013) ** 2))
        temp += ring.astype(np.float32) * fan_amp * 0.12

        temp = self._add_sensor_noise(temp, ambient)

        target_max = float(self.rng.uniform(*profile["max_temp"]))
        current_max = float(np.max(temp))
        if current_max > ambient + 1e-6:
            gain = (target_max - ambient) / (current_max - ambient)
            temp = ambient + (temp - ambient) * gain

        temp = np.clip(temp, 22.0, 95.0).astype(np.float32)

        region_masks = {
            "cpu": (cpu_blob > 0.35).astype(np.uint8) * 255,
            "gpu": (gpu_blob > 0.35).astype(np.uint8) * 255,
            "heatpipe": (heatpipe_blob > 0.24).astype(np.uint8) * 255,
            "keyboard": (keyboard_mask > 0.5).astype(np.uint8) * 255,
            "fan": (fan_blob > 0.35).astype(np.uint8) * 255,
            "palm_rest": np.pad(
                np.ones((self.height - palm_y0, self.width), dtype=np.uint8) * 255,
                ((palm_y0, 0), (0, 0)),
                mode="constant",
                constant_values=0,
            ),
        }

        train_mask = ((temp > (ambient + 8.0)) & (yy < self.height * 0.80)).astype(np.uint8) * 255

        hottest_flat_idx = int(np.argmax(temp))
        hot_y, hot_x = np.unravel_index(hottest_flat_idx, temp.shape)
        hotspot_coordinate = (float(hot_x), float(hot_y))
        hotspot_temperature = float(temp[hot_y, hot_x])

        thermal_rgb = self.render_flir(temp)

        top_hotspots = self._extract_top_hotspots(temp, hotspot_count)
        hotspot_centers = [(x, y) for x, y, _ in top_hotspots]
        hotspot_temps = [t for _, _, t in top_hotspots]

        return ThermalFrame(
            image=temp,
            mask=train_mask,
            centers=hotspot_centers,
            temperatures=hotspot_temps,
            width=self.width,
            height=self.height,
            thermal_image=thermal_rgb,
            region_masks=region_masks,
            hotspot_coordinate=hotspot_coordinate,
            hotspot_temperature=hotspot_temperature,
            workload=scene_workload,
        )

    def generate_batch(
        self,
        sample_count: int,
        workloads: Optional[Iterable[str]] = None,
    ) -> list[ThermalFrame]:
        """Generate many unique laptop thermal scenes, suitable for 100+ samples."""
        if sample_count <= 0:
            return []

        workload_list = list(workloads) if workloads is not None else list(self.WORKLOAD_PROFILES.keys())
        if not workload_list:
            workload_list = list(self.WORKLOAD_PROFILES.keys())

        frames: list[ThermalFrame] = []
        for _ in range(sample_count):
            selected = str(self.rng.choice(workload_list))
            frames.append(self.generate(workload=selected))
        return frames

    def render_flir(self, temperature_matrix: np.ndarray) -> np.ndarray:
        """Render a FLIR-like RGB thermal image from raw temperature matrix."""
        low = float(np.percentile(temperature_matrix, 2.0))
        high = float(np.percentile(temperature_matrix, 99.2))
        if high <= low:
            high = low + 1e-3

        norm = np.clip((temperature_matrix - low) / (high - low), 0.0, 1.0)
        norm_u8 = (norm * 255.0).astype(np.uint8)
        colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_INFERNO)

        # Mild sensor blur and vignette for camera-like rendering.
        colored = cv2.GaussianBlur(colored, (3, 3), 0.8)
        yy, xx = np.indices((self.height, self.width), dtype=np.float32)
        rr = np.sqrt((xx - self.width * 0.5) ** 2 + (yy - self.height * 0.5) ** 2)
        rr /= np.max(rr) + 1e-6
        vignette = np.clip(1.0 - 0.18 * rr, 0.80, 1.0)
        colored = np.clip(colored.astype(np.float32) * vignette[..., None], 0, 255).astype(np.uint8)

        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def _workload_from_hotspot_count(self, hotspot_count: int) -> str:
        """Keep backward compatibility with existing GUI hotspot slider."""
        map_by_count = {
            1: "light",
            2: "medium",
            3: "heavy",
            4: "cpu_stress",
            5: "dual_stress",
        }
        return map_by_count.get(int(np.clip(hotspot_count, 1, 5)), "medium")

    def _add_sensor_noise(self, temp: np.ndarray, ambient: float) -> np.ndarray:
        """Add realistic thermal sensor noise and dead pixels."""
        noisy = temp.copy()

        gaussian = self.rng.normal(0.0, self.noise_std, size=noisy.shape).astype(np.float32)
        low_freq = cv2.GaussianBlur(
            self.rng.normal(0.0, self.noise_std * 0.35, size=noisy.shape).astype(np.float32),
            (0, 0),
            sigmaX=self.width * 0.02,
            sigmaY=self.height * 0.02,
        )
        noisy += gaussian + low_freq

        # Small line-wise fluctuation similar to sensor readout drift.
        row_drift = self.rng.normal(0.0, 0.08, size=(self.height, 1)).astype(np.float32)
        noisy += row_drift

        dead_pixel_ratio = self.rng.uniform(0.0003, 0.0013)
        dead_count = max(1, int(self.width * self.height * dead_pixel_ratio))
        ys = self.rng.integers(0, self.height, dead_count)
        xs = self.rng.integers(0, self.width, dead_count)
        dead_delta = self.rng.choice([-1.0, 1.0], size=dead_count).astype(np.float32) * self.rng.uniform(
            2.0,
            7.0,
            size=dead_count,
        ).astype(np.float32)
        noisy[ys, xs] = np.clip(noisy[ys, xs] + dead_delta, ambient - 3.0, 98.0)

        return noisy

    def _extract_top_hotspots(self, temp: np.ndarray, hotspot_count: int) -> list[Tuple[float, float, float]]:
        """Extract top-N distinct hotspot peaks using simple non-maximum suppression."""
        count = int(np.clip(hotspot_count, 1, 5))
        work = temp.copy()
        peaks: list[Tuple[float, float, float]] = []

        suppress_r = max(6, int(min(self.width, self.height) * 0.035))

        for _ in range(count):
            flat_idx = int(np.argmax(work))
            y, x = np.unravel_index(flat_idx, work.shape)
            t = float(work[y, x])
            peaks.append((float(x), float(y), t))

            y0 = max(0, y - suppress_r)
            y1 = min(self.height, y + suppress_r + 1)
            x0 = max(0, x - suppress_r)
            x1 = min(self.width, x + suppress_r + 1)
            work[y0:y1, x0:x1] = -1e9

        return peaks

    @staticmethod
    def _gaussian_2d(
        xx: np.ndarray,
        yy: np.ndarray,
        cx: float,
        cy: float,
        sigma_x: float,
        sigma_y: float,
    ) -> np.ndarray:
        """Generate normalized anisotropic Gaussian field."""
        sx = max(float(sigma_x), 1.0)
        sy = max(float(sigma_y), 1.0)
        val = np.exp(-(((xx - cx) ** 2) / (2.0 * sx * sx) + ((yy - cy) ** 2) / (2.0 * sy * sy)))
        return val.astype(np.float32)

    def set_seed(self, seed: int) -> None:
        """Reset RNG seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
