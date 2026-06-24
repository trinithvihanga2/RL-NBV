import os
import numpy as np
import open3d as o3d
from envs.utils import camera_axes

PARTIAL_RENDER_WIDTH = 640
PARTIAL_RENDER_HEIGHT = 480
PARTIAL_RENDER_FX = 525.0
PARTIAL_RENDER_FY = 525.0
PARTIAL_RENDER_CX = 319.5
PARTIAL_RENDER_CY = 239.5
MODEL_NORMALIZATION_SCALE = 0.8

class EnvironmentRenderer:
    def __init__(self, data_path, shapenet_reader, orbit_radius, collision_check_samples, collision_penalty_weight, logger):
        self.data_path = data_path
        self.shapenet_reader = shapenet_reader
        self.orbit_radius = orbit_radius
        self.collision_check_samples = collision_check_samples
        self.collision_penalty_weight = collision_penalty_weight
        self.logger = logger
        self._mesh_cache = {}

    def _resolve_current_model_mesh_path(self):
        model_name = self.shapenet_reader.get_model_info()
        candidate_paths = [
            os.path.join(self.data_path, model_name, "model.obj"),
            os.path.join(self.data_path, f"{model_name}.obj"),
            os.path.join(self.shapenet_reader.data_path, model_name, "model.obj"),
            os.path.join(self.shapenet_reader.data_path, f"{model_name}.obj"),
        ]
        for candidate_path in candidate_paths:
            if os.path.isfile(candidate_path):
                return candidate_path
        return None

    def load_current_model_mesh(self):
        model_name = self.shapenet_reader.get_model_info()
        cached_mesh = self._mesh_cache.get(model_name)
        if cached_mesh is not None:
            return cached_mesh

        mesh_path = self._resolve_current_model_mesh_path()
        if mesh_path is None:
            self.logger.debug(
                "[continuous] No model.obj found for model %s; using canonical-point fallback",
                model_name,
            )
            return None

        mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)
        if mesh.is_empty() or not mesh.has_vertices():
            self.logger.warning(
                "[continuous] Failed to load mesh from %s; using canonical-point fallback",
                mesh_path,
            )
            return None

        mesh.compute_vertex_normals()
        vertices = np.asarray(mesh.vertices)
        centroid = vertices.mean(axis=0)
        mesh.translate(-centroid)
        vertices = np.asarray(mesh.vertices)
        max_dist = float(np.max(np.linalg.norm(vertices, axis=1)))
        if max_dist > 0.0:
            mesh.scale(MODEL_NORMALIZATION_SCALE / max_dist, center=(0, 0, 0))
        mesh.compute_vertex_normals()

        self._mesh_cache[model_name] = mesh
        self.logger.info(
            "[continuous] Loaded mesh for %s from %s", model_name, mesh_path
        )
        return mesh

    def _build_collision_scene(self, mesh):
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)
        return scene

    def check_transfer_collision(self, start_position, end_position, travel_time):
        mesh = self.load_current_model_mesh()
        if mesh is not None:
            try:
                scene = self._build_collision_scene(mesh)
                sample_count = max(
                    self.collision_check_samples,
                    int(np.ceil(max(float(travel_time), 1.0) * 8.0)),
                )
                fractions = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)[1:-1]
                if fractions.size == 0:
                    return False, 0.0, None

                samples = (
                    start_position[np.newaxis, :] * (1.0 - fractions[:, np.newaxis])
                    + end_position[np.newaxis, :] * fractions[:, np.newaxis]
                ).astype(np.float32)
                sample_tensor = o3d.core.Tensor(samples, dtype=o3d.core.Dtype.Float32)

                if hasattr(scene, "compute_signed_distance"):
                    signed_distance = scene.compute_signed_distance(
                        sample_tensor
                    ).numpy()
                    signed_distance = np.asarray(signed_distance, dtype=np.float32)
                    min_clearance = float(np.min(signed_distance))
                    collision_mask = signed_distance <= 0.0
                    if np.any(collision_mask):
                        penetration = max(0.0, -min_clearance)
                        collision_penalty = self.collision_penalty_weight * (
                            1.0 + penetration
                        )
                        return True, collision_penalty, min_clearance
                    return False, 0.0, min_clearance
            except Exception as exc:
                self.logger.debug("[COLLISION] signed-distance check failed: %s", exc)

        # Conservative fallback when signed distance is unavailable.
        samples = np.linspace(
            0.0, 1.0, max(self.collision_check_samples, 2), dtype=np.float32
        )[1:-1]
        if samples.size == 0:
            return False, 0.0, None
        interpolated = (
            start_position[np.newaxis, :] * (1.0 - samples[:, np.newaxis])
            + end_position[np.newaxis, :] * samples[:, np.newaxis]
        )
        radii = np.linalg.norm(interpolated, axis=1)
        collision_radius = float(MODEL_NORMALIZATION_SCALE)
        min_clearance = float(np.min(radii) - collision_radius)
        if np.any(radii <= collision_radius):
            penetration = max(0.0, -min_clearance)
            collision_penalty = self.collision_penalty_weight * (1.0 + penetration)
            return True, collision_penalty, min_clearance
        return False, 0.0, min_clearance

    def render_partial_points_from_mesh(self, mesh, position):
        scene = o3d.t.geometry.RaycastingScene()
        mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_t)

        right, up_vec, fwd, origin = camera_axes(position)

        us = np.arange(PARTIAL_RENDER_WIDTH, dtype=np.float64)
        vs = np.arange(PARTIAL_RENDER_HEIGHT, dtype=np.float64)
        uu, vv = np.meshgrid(us, vs)

        xc = (uu - PARTIAL_RENDER_CX) / PARTIAL_RENDER_FX
        yc = (vv - PARTIAL_RENDER_CY) / PARTIAL_RENDER_FY

        dirs = (
            fwd[np.newaxis, np.newaxis, :]
            + xc[..., np.newaxis] * right[np.newaxis, np.newaxis, :]
            - yc[..., np.newaxis] * up_vec[np.newaxis, np.newaxis, :]
        )

        norms = np.linalg.norm(dirs, axis=-1, keepdims=True)
        dirs_unit = (dirs / np.maximum(norms, 1e-12)).astype(np.float32)
        origins = np.full(dirs_unit.shape, origin, dtype=np.float32)

        rays = np.concatenate(
            [origins.reshape(-1, 3), dirs_unit.reshape(-1, 3)], axis=1
        )
        rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        result = scene.cast_rays(rays_t)
        t_hit = (
            result["t_hit"].numpy().reshape(PARTIAL_RENDER_HEIGHT, PARTIAL_RENDER_WIDTH)
        )

        valid = np.isfinite(t_hit) & (t_hit > 0.0)
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32)

        t_v = t_hit[valid].astype(np.float64)
        dirs_v = dirs_unit[valid].astype(np.float64)
        pts = origin + t_v[:, np.newaxis] * dirs_v

        keep = np.linalg.norm(pts, axis=1) < 1.2
        pts = pts[keep]
        return pts.astype(np.float32)

    def get_points_from_position(self, position, canonical_points):
        mesh = self.load_current_model_mesh()
        if mesh is not None:
            points = self.render_partial_points_from_mesh(mesh, position)
            if points.shape[0] > 0:
                return points

        canonical_points = np.asarray(canonical_points, dtype=np.float32)
        if canonical_points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)

        position = np.asarray(position, dtype=np.float32)
        d = float(np.linalg.norm(position))
        if d < 1e-12:
            return np.zeros((0, 3), dtype=np.float32)

        u_hat = position / d
        chief_radius = float(self.orbit_radius)
        rhs = (chief_radius * chief_radius) / d
        visible_mask = np.asarray(canonical_points @ u_hat >= rhs, dtype=bool)
        return canonical_points[visible_mask]
