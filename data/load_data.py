import pickle
import numpy as np
import pandas as pd
from ast import literal_eval
import os
import pathlib
import json

def trim_scans_per_referit3d_data(referit_data, scans):
    # remove scans not in referit_data
    in_r3d = referit_data.scan_id.unique()
    to_drop = []
    for k in scans:
        if k not in in_r3d:
            to_drop.append(k)
    for k in to_drop:
        del scans[k]
    print('Dropped {} scans to reduce mem-foot-print.'.format(len(to_drop)))
    return scans

def mean_color(scan_ids, all_scans):
    mean_rgb = np.zeros((1, 3), dtype=np.float32)
    n_points = 0
    for scan_id in scan_ids:
        color = all_scans[scan_id].color
        mean_rgb += np.sum(color, axis=0)
        n_points += len(color)
    mean_rgb /= n_points  #
    return mean_rgb

def load_filtered_data(pkl_scannet_file, refer_train, refer_val, add_pad=True):
    pkls = pkl_scannet_file.split(';')
    all_scans = dict()
    for pkl_f in pkls:
        with open(pkl_f, 'rb') as f:
            scans = pickle.load(f)
        scans = {scan.scan_id: scan for scan in scans}
        all_scans.update(scans)
    class_labels = set()
    for k, scan in all_scans.items():
        idx = np.array([o.object_id for o in scan.three_d_objects])
        class_labels.update([o.semantic_label(scan) for o in scan.three_d_objects])
        assert np.all(idx == np.arange(len(idx)))

    class_to_idx = {}
    i = 0
    for el in sorted(class_labels):
        class_to_idx[el] = i
        i += 1

    # Add the pad class needed for object classification
    if add_pad:
        class_to_idx['pad'] = len(class_to_idx)

    train_data = pd.read_csv(refer_train)
    val_data = pd.read_csv(refer_val)
    referit_data = pd.concat([train_data, val_data])
    referit_data = referit_data[['tokens', 'instance_type', 'scan_id', 'is_train',
                                 'dataset', 'target_id', 'utterance', 'stimulus_id']]
    referit_data.tokens = referit_data['tokens'].apply(literal_eval)
    return all_scans, class_to_idx, referit_data