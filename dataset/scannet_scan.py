import json
import os.path

import numpy as np
import os.path as osp
import warnings
from collections import defaultdict
import pandas as pd
from plyfile import PlyData

from .utils import invert_dictionary, read_dict
from .three_d_object import ThreeDObject
import random


class ScannetDataset(object):
    """
    Holds Scannet mesh and labels data paths and some needed class labels mappings
    Note: data downloaded from: http://www.scan-net.org/changelog#scannet-v2-2018-06-11
    """

    def __init__(self, top_scan_dir, idx_to_semantic_cls_file,
                 instance_cls_to_semantic_cls_file, axis_alignment_info_file):
        self.top_scan_dir = top_scan_dir

        self.idx2semantic_cls = read_dict(idx_to_semantic_cls_file)

        self.semantic_cls2idx = invert_dictionary(self.idx2semantic_cls)

        self.instance_cls2semantic_cls = read_dict(instance_cls_to_semantic_cls_file)

        self.semantic_cls2instance_cls = defaultdict(list)

        for k, v in self.instance_cls2semantic_cls.items():
            self.semantic_cls2instance_cls[v].append(k)

        if axis_alignment_info_file:
            self.scans_axis_alignment_matrices = read_dict(axis_alignment_info_file)


class ScannetScan(object):
    """
    Keep track of the point-cloud associated with the scene of Scannet. Includes meta-information such as the
    object that exist in the scene, their semantic labels and their RGB color.
    """

    def __init__(self, scan_id, scannet_dataset, apply_global_alignment=True, load_semantic_label=True, sample=0):
        """
            :param scan_id: (string) e.g. 'scene0705_00'
            :scannet_dataset: (ScannetDataset) captures the details about the class-names, top-directories etc.
        """
        self.dataset = scannet_dataset
        self.scan_id = scan_id
        self.pc, self.semantic_label, self.color = \
            self.load_point_cloud_with_meta_data(apply_global_alignment=apply_global_alignment,
                                                 load_semantic_label=load_semantic_label)
        self.sample = False
        self.selected_pis = list(range(0, len(self.pc)))
        if sample > 0 and len(self.pc) > sample:
            self.sample = True
            selected_pis = sorted(random.sample(list(range(0, len(self.pc))), sample))
            self.pc = self.pc[selected_pis]
            self.color = self.color[selected_pis]
            if self.semantic_label is not None:
                self.semantic_label = self.semantic_label[selected_pis]
            self.selected_pis = selected_pis

        self.three_d_objects = None  # A list with ThreeDObject contained in this Scan

        self.target_boxes = None

    def __str__(self, verbose=True):
        res = '{}'.format(self.scan_id)
        if verbose:
            res += ' with {} points'.format(self.n_points())
        return res

    def n_points(self):
        return len(self.pc)

    def verify_read_data_correctness(self, scan_aggregation, segment_file, segment_indices):
        c1 = scan_aggregation['sceneId'][len('scannet.'):] == self.scan_id
        scan_segs_suffix = '_vh_clean_2.0.010000.segs.json'
        segment_dummy = self.scan_id + scan_segs_suffix
        c2 = segment_file == segment_dummy
        c3 = len(segment_indices) == self.n_points() or len(self.selected_pis)==self.n_points()
        c = np.array([c1, c2, c3])
        if not np.all(c):
            warnings.warn('{} has some issue'.format(self.scan_id))
        return c

    def load_point_cloud_with_meta_data(self, load_semantic_label=True, load_color=True, apply_global_alignment=True):
        """
        :param load_semantic_label:
        :param load_color:
        :param apply_global_alignment: rotation/translation of scan according to Scannet meta-data.
        :return:
        """
        scan_ply_suffix = '_vh_clean_2.labels.ply'
        mesh_ply_suffix = '_vh_clean_2.ply'

        scan_data_file = osp.join(self.dataset.top_scan_dir, self.scan_id, self.scan_id + mesh_ply_suffix)
        data = PlyData.read(scan_data_file)
        x = np.asarray(data.elements[0].data['x'])
        y = np.asarray(data.elements[0].data['y'])
        z = np.asarray(data.elements[0].data['z'])
        pc = np.stack([x, y, z], axis=1)

        color = None
        if load_color:
            r = np.asarray(data.elements[0].data['red'])
            g = np.asarray(data.elements[0].data['green'])
            b = np.asarray(data.elements[0].data['blue'])
            color = (np.stack([r, g, b], axis=1) / 256.0).astype(np.float32)

        label = None
        if load_semantic_label:
            scan_data_file = osp.join(self.dataset.top_scan_dir, self.scan_id, self.scan_id + scan_ply_suffix)
            data = PlyData.read(scan_data_file)
            label = np.asarray(data.elements[0].data['label'])

        # Global alignment of the scan
        if apply_global_alignment:
            pc = self.align_to_axes(pc)

        return pc, label, color

    def load_point_clouds_of_all_objects(self, exclude_instances=None):
        if self.sample:
            pc_map = {p: i for i, p in enumerate(self.selected_pis)}

        scan_aggregation_suffix = '.aggregation.json'
        aggregation_file = osp.join(self.dataset.top_scan_dir, self.scan_id, self.scan_id + scan_aggregation_suffix)
        with open(aggregation_file) as fin:
            scan_aggregation = json.load(fin)

        scan_segs_suffix = '_vh_clean_2.0.010000.segs.json'
        segment_file = self.scan_id + scan_segs_suffix

        segments_file = osp.join(self.dataset.top_scan_dir, self.scan_id, segment_file)

        with open(segments_file) as fin:
            segments_info = json.load(fin)
            segment_indices = segments_info['segIndices']

        segment_dummy = scan_aggregation['segmentsFile'][len('scannet.'):]

        check = self.verify_read_data_correctness(scan_aggregation, segment_dummy, segment_indices)

        segment_indices_dict = defaultdict(list)
        for i, s in enumerate(segment_indices):
            segment_indices_dict[s].append(i)  # Add to each segment, its point indices

        # iterate over every object
        all_objects = []
        for object_info in scan_aggregation['segGroups']:
            object_instance_label = object_info['label']
            object_id = object_info['objectId']
            if exclude_instances is not None:
                if object_instance_label in exclude_instances:
                    continue

            segments = object_info['segments']
            pc_loc = []
            # Loop over the object segments and get the all point indices of the object
            for s in segments:
                pc_loc.extend(segment_indices_dict[s])
            object_pc = pc_loc
            if self.sample:
                object_pc = [pc_map[pc] for pc in object_pc if pc in self.selected_pis]
            all_objects.append(ThreeDObject(object_id, object_pc, object_instance_label))
        self.three_d_objects = all_objects
        return check

    def instance_occurrences(self):
        """
        :return: (dict) instance_type (string) -> number of occurrences in the scan (int)
        """
        res = defaultdict(int)
        for o in self.three_d_objects:
            res[o.instance_label] += 1
        return res

    def align_to_axes(self, point_cloud):
        """
        Align the scan to xyz axes using the alignment matrix found in scannet.
        """
        # Get the axis alignment matrix
        alignment_matrix = self.dataset.scans_axis_alignment_matrices[self.scan_id]
        alignment_matrix = np.array(alignment_matrix, dtype=np.float32).reshape(4, 4)

        # Transform the points
        pts = np.ones((point_cloud.shape[0], 4), dtype=point_cloud.dtype)
        pts[:, 0:3] = point_cloud
        point_cloud = np.dot(pts, alignment_matrix.transpose())[:, :3]  # Nx4

        # Make sure no nans are introduced after conversion
        assert (np.sum(np.isnan(point_cloud)) == 0)

        return point_cloud
