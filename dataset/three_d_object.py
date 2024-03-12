import numpy as np
from .cuboid import OrientedCuboid


class ThreeDObject(object):
    """
    Representing a ScanNet 3D Object
    """

    def __init__(self, object_id, points, instance_label):
        self.object_id = object_id
        self.points = points
        self.instance_label = instance_label

        self.axis_aligned_bbox = None
        self.is_axis_aligned_bbox_set = False

        self.object_aligned_bbox = None
        self.has_object_aligned_bbox = False

        self.front_direction = None
        self.has_front_direction = False
        self._use_true_instance = True

        self.pc = None  # The point cloud (xyz)
        self.normalized_pc = None  # The normalized point cloud (xyz) in unit sphere
        self.color = None  # The point cloud (RGB) values

    def get_axis_align_bbox(self, normalize_pc=False):
        if self.is_axis_aligned_bbox_set:
            pass
        else:
            if normalize_pc:
                pc = self.normalized_pc
            else:
                pc = self.pc
            cx, cy, cz = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2.0
            lx, ly, lz = np.max(pc, axis=0) - np.min(pc, axis=0)
            rot = np.eye(N=3)
            assert (lx > 0 and ly > 0 and lz > 0)

            self.axis_aligned_bbox = OrientedCuboid(cx, cy, cz, lx, ly, lz, rot)  # 得到点云的box坐标
            self.is_axis_aligned_bbox_set = True
        return self.axis_aligned_bbox

    #
    def normalize_pc(self):
        """
        Normalize the object's point cloud to a unit sphere centered at the origin point
        """
        assert (self.pc is not None)
        point_set = self.pc - np.expand_dims(np.mean(self.pc, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        self.normalized_pc = point_set / dist  # scale

    def get_pc(self, scans, normalize=False):
        # Set the pc if not previously initialized
        if self.pc is None:
            self.pc = scans.pc[self.points]
        if normalize and self.normalized_pc is None:
            self.normalize_pc()

        if normalize:
            return self.normalized_pc
        return self.pc

    def get_color(self, scans):
        if self.color is None:
            self.color = scans.color[self.points]
        return self.color

    #
    def get_bbox(self, axis_aligned=False):
        """if you have object-align return this, else compute/return axis-aligned"""
        if not axis_aligned and self.has_object_aligned_bbox:
            return self.object_aligned_bbox
        else:
            return self.get_axis_align_bbox()

    #
    def semantic_label(self, scan):
        one_point = scan.semantic_label[self.points[0]]
        return scan.dataset.idx2semantic_cls[str(one_point)]

    def sample(self, n_samples, scans, normalized_pc=False):
        """sub-sample its pointcloud and color"""

        xyz = self.get_pc(scans, normalize=normalized_pc)
        color = self.get_color(scans)

        n_points = len(self.points)
        assert xyz.shape[0] == len(self.points)
        idx = np.random.choice(n_points, n_samples, replace=n_points < n_samples)

        res = {
            'xyz': xyz[idx],
            'color': color[idx]}

        return res
