"""
File: surfslam/mapping/frame_tracker.py

FrameTracker: A lightweight alternative to Mapper that defers pose association.

This class:
1. Receives completed frames one at a time and immediately stores depth/color to CPU
2. At the end of the run, receives final optimized poses and accumulates maps
3. Optionally publishes provisional Gaussian snapshots while keyframes arrive

Copyright 2023, Ford Center for Autonomous Vehicles at University of Michigan
All Rights Reserved.
"""

import time
from typing import Optional, Dict, Any

import torch
import torch.multiprocessing as mp
import numpy as np
from pathlib import Path

from kornia.geometry.depth import depth_to_3d_v2
import open3d as o3d
from common.frame import Frame
from common.settings import Settings
from common.signals import Signal, Slot, StopSignal
from common.shared_state import SharedState

from mapping.pointcloud_registration.pointcloud_cleaner import GPUPointCloudCleaner
from mapping.gaussian_map import GaussianMapWriter


def _extract_frame_stereo_timestamp(frame_item: Frame) -> float:
    """Extract timestamp from a Frame's stereo image."""
    stereo_candidate = getattr(frame_item, "stereo_image", None)
    return getattr(stereo_candidate, "timestamp")

def filter_points_above_water(pts3d_cam, cam_pose, water_surface):
    """Filter points that are above the water surface."""
    if water_surface is None or len(pts3d_cam) == 0:
        return np.ones(len(pts3d_cam), dtype=bool)

    pts3d_world = (cam_pose[:3, :3] @ pts3d_cam.T).T + cam_pose[:3, 3]

    surface_point = water_surface['point']
    surface_normal = water_surface['normal']
    threshold = water_surface['threshold']

    distances = np.dot(pts3d_world - surface_point, surface_normal)

    valid_mask = distances < -threshold

    return valid_mask

def estimate_water_surface_plane_in_camera(
    baro_data,
    imu_data,
    T_world_cam,
    surface_threshold=0.5,
    pressure_units="hpa",
    rho_water_kg_m3=997.0,
    g_m_s2=9.81,
    atm_pressure=986.3999633789062,
    R_cam_imu=None,
):
    if baro_data is None or imu_data is None:
        return None

    pressure_meas = float(baro_data["pressure"])
    pressure_atm = float(atm_pressure)  # must match baro units

    if pressure_units.lower() == "hpa":
        pressure_delta_pa = (pressure_meas - pressure_atm) * 100.0
    elif pressure_units.lower() == "pa":
        pressure_delta_pa = (pressure_meas - pressure_atm)
    else:
        raise ValueError("pressure_units must be 'hpa' or 'pa'.")

    depth_m = pressure_delta_pa / (rho_water_kg_m3 * g_m_s2)
    if not np.isfinite(depth_m):
        return None
    depth_m = float(max(0.0, depth_m))

    accel_imu = np.array(
        [imu_data["accel_x"], imu_data["accel_y"], imu_data["accel_z"]],
        dtype=np.float64,
    )
    accel_norm = float(np.linalg.norm(accel_imu))
    if accel_norm < 1e-9 or not np.isfinite(accel_norm):
        return None

    accel_dir_imu = accel_imu / accel_norm

    gravity_dir_imu = -accel_dir_imu

    if R_cam_imu is None:
        # TODO: onur get the extrinsics from the config file
        R_cam_imu = np.array(
            [[0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    else:
        R_cam_imu = np.asarray(R_cam_imu, dtype=np.float64)
        if R_cam_imu.shape != (3, 3):
            raise ValueError("R_cam_imu must be 3x3.")

    gravity_dir_cam = R_cam_imu @ gravity_dir_imu
    gravity_dir_cam /= np.linalg.norm(gravity_dir_cam)

    T_world_cam = np.asarray(T_world_cam, dtype=np.float64)
    if T_world_cam.shape != (4, 4):
        raise ValueError("T_world_cam must be 4x4.")

    R_world_cam = T_world_cam[:3, :3]
    p_world_cam = T_world_cam[:3, 3]

    gravity_dir_world = R_world_cam @ gravity_dir_cam
    gravity_dir_world /= np.linalg.norm(gravity_dir_world)

    n_world = -gravity_dir_world
    n_world /= np.linalg.norm(n_world)

    if float(np.dot(n_world, gravity_dir_world)) > 0.0:
        n_world = -n_world
    p0_world = p_world_cam + depth_m * n_world
    signed_height_check = float(
        np.dot(p0_world - p_world_cam, n_world))

    R_cam_world = R_world_cam.T
    n_cam = R_cam_world @ n_world
    n_cam /= np.linalg.norm(n_cam)

    p0_cam = depth_m * n_cam
    d_cam = -float(np.dot(n_cam, p0_cam))

    return {
        "point": p0_world,
        "normal": n_world,
        "depth": depth_m,
        "threshold": float(surface_threshold),
        "signed_height_check": signed_height_check,
        "plane_normal_cam": n_cam,
        "plane_offset_d_cam": d_cam,
        "point_on_plane_cam": p0_cam,
    }


class FrameTracker:
    """FrameTracker: Lightweight mapping module that defers pose association.

    This class:
    - Receives one frame at a time and immediately moves depth/color to CPU
    - Stores minimal data indexed by keyframe ID
    - At shutdown, receives final poses and accumulates the selected map
    - Optionally publishes provisional Gaussian snapshots from current poses
    """

    def __init__(
        self,
        settings: Settings,
        final_poses_signal: Signal,
        completed_frame_signal: Signal,
        backend_to_mapper_signal: Signal,
    ):
        """
        Args:
            settings: Top level settings for the entire SLAM module
            final_poses_signal: Signal to receive final poses at the end of the run
                               Format: dict of {keyframe_id: 4x4 numpy pose matrix}
            completed_frame_signal: Signal to receive completed frames with depth (one at a time)
            backend_to_mapper_signal: Signal to receive latest backend measurements (IMU, DVL, Barometer)
        """
        self._settings = settings

        # Keep the existing point-cloud output as the default.  Gaussian maps
        # reuse the same keyframes, depth, colors, and optimized poses, so the
        # representation can be selected without changing the SLAM frontend or
        # pose-graph backend.
        self._map_type = str(self._settings.mapper.get("map_type", "pointcloud")).lower()
        if self._map_type not in {"pointcloud", "gaussian"}:
            raise ValueError(
                "mapper.map_type must be either 'pointcloud' or 'gaussian', "
                f"got {self._map_type!r}"
            )
        self._gaussian_config = self._settings.mapper.get("gaussian", {})
        self._gaussian_mode = str(
            self._gaussian_config.get("mode", "final")
        ).lower()
        if self._gaussian_mode not in {"final", "online"}:
            raise ValueError(
                "mapper.gaussian.mode must be either 'final' or 'online', "
                f"got {self._gaussian_mode!r}"
            )
        self._gaussian_writer = GaussianMapWriter(self._gaussian_config)
        self._online_gaussian_enabled = (
            self._map_type == "gaussian" and self._gaussian_mode == "online"
        )
        self._online_update_interval = max(
            1, int(self._gaussian_config.get("online_update_interval_keyframes", 5))
        )
        self._online_gaussian_points = []
        self._online_gaussian_colors = []
        self._online_gaussian_keyframes = 0

        self._final_poses_slot: Slot = final_poses_signal.register()
        self._completed_frame_slot: Slot = completed_frame_signal.register()
        self._backend_data_slot: Slot = backend_to_mapper_signal.register()

        self._processed_stop_signal = mp.Value("i", 0)
        self._term_signal = mp.Value("i", 0)

        # Store frame data indexed by keyframe ID (depth/color on CPU)
        # Structure: {keyframe_id: {"depth_image": Tensor (CPU), "color_image": Tensor (CPU), "timestamp": float, "barometer": dict, "imu": dict}}
        self._keyframe_data: Dict[int, Dict[str, Any]] = {}

        # Point cloud cleaner (optional)
        if self._settings.mapper.registration.cleanup.enabled:
            self._pc_cleaner = GPUPointCloudCleaner(
                self._settings.mapper.registration.cleanup,
                device="cuda:0",
            )
        else:
            self._pc_cleaner = None
        
        # Buffers for water surface estimation
        self._barometer_buffer = []  # List of {'timestamp': float, 'pressure': float}
        self._imu_buffer = []  # List of {'timestamp': float, 'accel_x': float, 'accel_y': float, 'accel_z': float}
        self._buffer_max_size = 100  # Keep last N readings
        
        # Water surface filtering settings
        self._enable_water_filtering = self._settings.mapper.get('enable_water_surface_filtering', True)
        self._water_surface_threshold = self._settings.mapper.get('water_surface_threshold', 0.5)
        print(f"Water surface filtering: {'enabled' if self._enable_water_filtering else 'disabled'}")
            
        print('FrameTracker Initialized!')

    def _get_pc(
        self, depth_image: torch.Tensor, intrinsics: torch.Tensor, color: Optional[torch.Tensor]
    ) -> tuple:
        """Convert depth image to point cloud in camera frame."""
        initial_pc = depth_to_3d_v2(
            depth_image,
            intrinsics.unsqueeze(0).cuda(),
        ).squeeze().reshape(-1, 3)
        color_flat = color.permute(1, 2, 0).reshape(-1, 3).cuda() if color is not None else None
        
        if self._pc_cleaner is not None:
            clean_pc, clean_color = self._pc_cleaner.clean(initial_pc, color_flat)
            return clean_pc, clean_color
        else:
            return initial_pc, color_flat

    def _se3_apply_to_points(self, world_T_cam: np.ndarray, points_cam_np: np.ndarray) -> np.ndarray:
        """Apply SE3 transform to points."""
        R = world_T_cam[:3, :3]
        t = world_T_cam[:3, 3]
        return (points_cam_np @ R.T) + t[None, :]

    def start(self):
        """Initialize the frame tracker."""
        print("FrameTracker spawned and started.")
        self._keyframe_data = {}
        self._online_gaussian_points = []
        self._online_gaussian_colors = []
        self._online_gaussian_keyframes = 0

    def _find_closest_reading(self, target_timestamp: float, buffer: list, key: str = None) -> Optional[dict]:
        """Find the reading in buffer closest to target_timestamp."""
        if not buffer:
            return None
        
        min_diff = float('inf')
        closest = None
        
        for reading in buffer:
            diff = abs(reading['timestamp'] - target_timestamp)
            if diff < min_diff:
                min_diff = diff
                closest = reading
        
        return closest
    
    def _process_single_frame(self, completed_frame: Frame) -> bool:
        """Process a single completed frame and store its data.
        
        Returns True if frame was stored, False if skipped.
        """
        keyframe_id = completed_frame.get_id()
        
        depth_image = completed_frame.depth_image.image if completed_frame.depth_image else None
        color_image = completed_frame.stereo_image.left_image if completed_frame.stereo_image else None
        timestamp = _extract_frame_stereo_timestamp(completed_frame)
        
        if depth_image is None:
            print(f"FrameTracker: Skipped frame {keyframe_id} (no depth image)")
            return False
        
        # BGR to RGB conversion for color
        if color_image is not None:
            color_image = color_image[[2, 1, 0], :, :]
        
        # Find closest barometer and IMU readings for water surface filtering
        closest_baro = self._find_closest_reading(timestamp, self._barometer_buffer)
        closest_imu = self._find_closest_reading(timestamp, self._imu_buffer)

        # Move to CPU immediately to free GPU memory and avoid serialization issues
        self._keyframe_data[keyframe_id] = {
            "depth_image": depth_image.cpu(),
            "color_image": color_image.cpu() if color_image is not None else None,
            "timestamp": float(timestamp),
            "barometer": closest_baro,
            "imu": closest_imu,
        }

        if self._online_gaussian_enabled:
            try:
                self._update_online_gaussian_map(
                    completed_frame, self._keyframe_data[keyframe_id]
                )
            except Exception as exc:
                # A provisional visualization must never take down the SLAM
                # pipeline.  The final Gaussian/point-cloud export remains
                # available because the raw keyframe was already stored.
                print(f"FrameTracker: disabling online Gaussian map: {exc}")
                self._online_gaussian_enabled = False
        return True

    def _update_online_gaussian_map(
        self, completed_frame: Frame, data: Dict[str, Any]
    ) -> None:
        """Add one keyframe using its provisional pose and publish snapshots.

        The pose is attached by ``Tracker`` before the frame is sent through
        the stereo/depth pipeline.  Loop closures can revise it later, so this
        provisional map is for live visualization; ``stop()`` still rebuilds
        the final map from optimized poses.
        """
        camera_pose = completed_frame.get_camera_pose()
        if camera_pose is None:
            return

        world_T_cam = camera_pose.get_transformation_matrix().detach().cpu().numpy()
        intrinsics: torch.Tensor = self._settings.calibration["camera_intrinsic"]["k"]
        depth_image = data["depth_image"].cuda()
        color_image = (
            data["color_image"].cuda()
            if data["color_image"] is not None
            else None
        )
        pc_cam, pc_color = self._get_pc(depth_image, intrinsics, color_image)
        pc_cam_np = pc_cam.cpu().numpy()

        if self._enable_water_filtering:
            baro_data = data.get("barometer")
            imu_data = data.get("imu")
            if baro_data is not None and imu_data is not None:
                water_surface = estimate_water_surface_plane_in_camera(
                    baro_data,
                    imu_data,
                    world_T_cam,
                    surface_threshold=self._water_surface_threshold,
                    pressure_units="hpa",
                )
                if water_surface is not None:
                    valid_mask = filter_points_above_water(
                        pc_cam_np, world_T_cam, water_surface
                    )
                    pc_cam_np = pc_cam_np[valid_mask]
                    if pc_color is not None:
                        pc_color = torch.from_numpy(pc_color.cpu().numpy()[valid_mask])

        if pc_cam_np.size == 0:
            return

        pc_world_np = self._se3_apply_to_points(world_T_cam, pc_cam_np)
        if pc_color is None:
            colors_np = np.tile(
                np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
                (pc_world_np.shape[0], 1),
            )
        else:
            colors_np = pc_color.cpu().float().numpy() / 255.0

        self._online_gaussian_points.append(pc_world_np)
        self._online_gaussian_colors.append(colors_np)
        self._online_gaussian_keyframes += 1

        if self._online_gaussian_keyframes % self._online_update_interval != 0:
            return

        self._write_online_gaussian_map()

    def _write_online_gaussian_map(self) -> None:
        if not self._online_gaussian_points:
            return

        points = np.vstack(self._online_gaussian_points)
        colors = np.vstack(self._online_gaussian_colors)
        output_path = Path(self._settings.log_directory) / "reconstructed_gaussians_live.ply"
        result = self._gaussian_writer.write(output_path, points, colors)

        # Keep the already reduced map as the next accumulator so online
        # memory stays bounded by max_gaussians rather than raw pixels.
        self._online_gaussian_points = [result["points"]]
        self._online_gaussian_colors = [result["colors"]]
        print(
            f"FrameTracker: Online Gaussian snapshot with {result['count']} "
            f"primitives -> {result['ply_path']}"
        )

    def update(self):
        """Process incoming frames and backend data."""
        # Process backend data if available
        if self._backend_data_slot.has_value():
            backend_data = self._backend_data_slot.get_value()
            if backend_data is not None and not isinstance(backend_data, StopSignal):
                imu = backend_data.get("imu")
                baro = backend_data.get("barometer")
                
                # Store barometer data
                if baro is not None:
                    self._barometer_buffer.append({
                        'timestamp': float(baro.timestamp),
                        'pressure': float(baro.pressure)
                    })
                    if len(self._barometer_buffer) > self._buffer_max_size:
                        self._barometer_buffer.pop(0)
                
                # Store IMU data
                if imu is not None:
                    self._imu_buffer.append({
                        'timestamp': float(imu.timestamp),
                        'accel_x': float(imu.ax),
                        'accel_y': float(imu.ay),
                        'accel_z': float(imu.az)
                    })
                    if len(self._imu_buffer) > self._buffer_max_size:
                        self._imu_buffer.pop(0)
            

        # Process only one frame per update call to avoid queue buildup
        if not self._completed_frame_slot.has_value():
            return

        frame = self._completed_frame_slot.get_value()

        if frame is None:
            return

        if isinstance(frame, StopSignal):
            self.stop()
            return

        tic = time.perf_counter()
        stored = self._process_single_frame(frame)
        toc = time.perf_counter()


    def _receive_final_poses(self) -> Dict[int, np.ndarray]:
        """Wait for and receive final poses from the signal.
        
        Returns dict of {keyframe_id: 4x4 numpy pose matrix}
        """
        print("FrameTracker: Waiting for final poses...")
        
        final_poses = {}
        max_wait_time = 30.0  # Maximum seconds to wait for poses
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            if self._final_poses_slot.has_value():
                pose_data = self._final_poses_slot.get_value()
                
                if pose_data is None:
                    continue
                    
                if isinstance(pose_data, StopSignal):
                    break
                    
                # Expect pose_data to be a dict of {keyframe_id: pose_matrix}
                if isinstance(pose_data, dict):
                    for kf_id, pose_val in pose_data.items():
                        # Extract the 4x4 numpy matrix
                        if hasattr(pose_val, "_transformation_matrix"):
                            pose_matrix = pose_val._transformation_matrix.cpu().numpy()
                        elif isinstance(pose_val, torch.Tensor):
                            pose_matrix = pose_val.cpu().numpy()
                        else:
                            pose_matrix = np.array(pose_val)
                        final_poses[int(kf_id)] = pose_matrix
                    
                    print(f"FrameTracker: Received {len(pose_data)} poses (total: {len(final_poses)})")
                    break  # Got poses, exit loop
            else:
                time.sleep(0.1)
        
        if not final_poses:
            print(f"FrameTracker: Warning - no poses received after {max_wait_time}s")
        
        return final_poses

    def stop(self, final_poses: Optional[Dict[int, np.ndarray]] = None):
        """Stop the frame tracker and save the accumulated map.
        
        Args:
            final_poses: Optional dict of {keyframe_id: 4x4 pose matrix}.
                        If provided, uses these directly instead of waiting for signal.
                        Useful for single-threaded mode.
        """
        print("FrameTracker stopping...")
        print(f"FrameTracker: Have {len(self._keyframe_data)} frames stored.")
        
        # Use provided poses or wait for them from the signal
        if final_poses is None:
            final_poses = self._receive_final_poses()
        else:
            print(f"FrameTracker: Using {len(final_poses)} directly provided poses.")
        
        if final_poses:
            output_dir = Path(self._settings.log_directory)
            if self._online_gaussian_enabled and self._online_gaussian_points:
                try:
                    self._write_online_gaussian_map()
                except Exception as exc:
                    print(f"FrameTracker: final online Gaussian snapshot failed: {exc}")
            if self._map_type == "gaussian":
                output_path = output_dir / "reconstructed_gaussians.ply"
            else:
                output_path = output_dir / "reconstructed_map.ply"
            self.save_map(str(output_path), final_poses)
            print(f"Saved {self._map_type} map to {output_path}.")
        else:
            print("FrameTracker: No poses received, cannot save map.")
        
        self._processed_stop_signal.value = 1

    def run(self, shared_state: SharedState) -> None:
        """Main loop for multiprocessing mode."""
        self.start()

        while not self._processed_stop_signal.value:
            self.update()
            time.sleep(0.001)

        print("FrameTracker Done. Waiting to terminate.")
        # Wait until an external terminate signal has been sent.
        while not self._term_signal.value:
            continue
        print("Exiting FrameTracker process.")

    def save_map(self, filepath: str, final_poses: Dict[int, np.ndarray]) -> None:
        """
        Save the selected map representation to disk.
        
        This is where the final pose association happens:
        1. For each keyframe, get its pose and depth image
        2. Transform point clouds to world frame
        3. Accumulate and save
        
        Args:
            filepath: Path to save the point cloud
            final_poses: Dict of {keyframe_id: 4x4 numpy pose matrix}
        """
        aggregated_points, aggregated_colors = self._accumulate_world_points(final_poses)
        if aggregated_points is None:
            return

        if self._map_type == "gaussian":
            result = self._gaussian_writer.write(
                filepath,
                aggregated_points,
                aggregated_colors,
            )
            print(
                f"FrameTracker: Saved {result['count']} Gaussian primitives to "
                f"{result['ply_path']} (parameters: {result['npz_path']})."
            )
            return

        # Create Open3D point cloud and save
        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(aggregated_points)
        pc_o3d.colors = o3d.utility.Vector3dVector(aggregated_colors)

        o3d.io.write_point_cloud(filepath, pc_o3d)
        print(f"FrameTracker: Saved {aggregated_points.shape[0]} points to {filepath}.")

    def _accumulate_world_points(
        self, final_poses: Dict[int, np.ndarray]
    ) -> tuple:
        """Build the world-frame RGB point set shared by both map backends."""
        if not self._keyframe_data:
            print("FrameTracker: No keyframe data to save.")
            return None, None

        print(
            f"FrameTracker: Accumulating map from {len(self._keyframe_data)} keyframes..."
        )
        print(f"FrameTracker: Have {len(final_poses)} poses available.")

        intrinsics: torch.Tensor = self._settings.calibration["camera_intrinsic"]["k"]
        all_points = []
        all_colors = []
        processed_count = 0
        skipped_count = 0

        for keyframe_id, data in sorted(self._keyframe_data.items()):
            # Python keyframe IDs are 0-indexed; the backend uses 1-indexed IDs.
            backend_kf_id = keyframe_id + 1
            if backend_kf_id not in final_poses:
                print(
                    f"FrameTracker: No pose for keyframe {keyframe_id} "
                    f"(backend ID {backend_kf_id}), skipping."
                )
                skipped_count += 1
                continue

            world_T_cam = final_poses[backend_kf_id]
            if world_T_cam is None:
                print(f"FrameTracker: Pose is None for keyframe {keyframe_id}, skipping.")
                skipped_count += 1
                continue

            depth_image = data["depth_image"].cuda()
            color_image = (
                data["color_image"].cuda()
                if data["color_image"] is not None
                else None
            )
            pc_cam, pc_color = self._get_pc(depth_image, intrinsics, color_image)
            pc_cam_np = pc_cam.cpu().numpy()

            if self._enable_water_filtering:
                baro_data = data.get("barometer")
                imu_data = data.get("imu")
                if baro_data is not None and imu_data is not None:
                    water_surface = estimate_water_surface_plane_in_camera(
                        baro_data,
                        imu_data,
                        world_T_cam,
                        surface_threshold=self._water_surface_threshold,
                        pressure_units="hpa",
                    )
                    if water_surface is not None:
                        points_before = len(pc_cam_np)
                        valid_mask = filter_points_above_water(
                            pc_cam_np, world_T_cam, water_surface
                        )
                        points_below = np.sum(valid_mask)
                        points_above = points_before - points_below
                        ratio = 100 * points_above / max(points_before, 1)
                        print(
                            f"Frame {keyframe_id}: {points_before} points total, "
                            f"{points_above} above water ({ratio:.1f}%), "
                            f"{points_below} below water."
                        )
                        pc_cam_np = pc_cam_np[valid_mask]
                        if pc_color is not None:
                            pc_color_np = pc_color.cpu().numpy()[valid_mask]
                            pc_color = torch.from_numpy(pc_color_np)

            if pc_cam_np.size == 0:
                continue

            pc_world_np = self._se3_apply_to_points(world_T_cam, pc_cam_np)
            all_points.append(pc_world_np)
            if pc_color is not None:
                all_colors.append(pc_color.cpu().float().numpy() / 255.0)
            else:
                all_colors.append(
                    np.tile(np.array([[0.5, 0.5, 0.5]]), (pc_cam_np.shape[0], 1))
                )
            processed_count += 1

        print(f"FrameTracker: Processed {processed_count} keyframes, skipped {skipped_count}")
        if not all_points:
            print("FrameTracker: No points accumulated, nothing to save.")
            return None, None

        return np.vstack(all_points), np.vstack(all_colors)
