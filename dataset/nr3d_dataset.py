import logging

import numpy as np
from torch.utils.data import Dataset
from functools import partial
from dataset.utils import *
from dataset.transform import mean_rgb_unit_norm_transform
from torch.utils.data import DataLoader


class Nr3dDataset(Dataset):
    def __init__(self, references, scans, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None, mapping=None, visualization=False):

        self.references = references
        self.scans = scans
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        self.mapping = mapping
        if not check_segmented_object_order(scans):
            raise ValueError

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan_id = ref['scan_id']
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        ori_tokens = ref['tokens']
        tokens = " ".join(ori_tokens)

        related_class = set()
        for tok in ori_tokens:
            if tok in self.mapping:
                related_class.add(self.mapping[tok])
            elif tok[:-1] in self.mapping:
                related_class.add(self.mapping[tok[:-1]])
            elif tok.endswith('es') and tok[:-2] in self.mapping:
                related_class.add(self.mapping[tok[:-2]])

        return scan, target, tokens, scan_id, related_class

    def prepare_distractors(self, scan, target, related_class):
        target_label = target.semantic_label(scan)
        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.semantic_label(scan) == target_label and (o != target))]
        # Then all more objects up to max-number of distractors
        already_included = [target_label]

        relaters = [o for o in scan.three_d_objects if
                    (o.semantic_label(scan) not in already_included and o.semantic_label(scan) in related_class)]
        already_included = already_included + list(related_class)
        np.random.shuffle(relaters)
        clutter = [o for o in scan.three_d_objects if o.semantic_label(scan) not in already_included]

        distractors.extend(relaters)
        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors

    def __getitem__(self, index):
        res = dict()
        scan, target, tokens, scan_id, related_class = self.get_reference_data(index)
        res['mention_class'] = len(related_class)
        # Make a context of distractors
        context = self.prepare_distractors(scan, target, related_class)
        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array([sample_scan_object(o, scan, self.points_per_object) for o in context])

        # mark their classes
        box_info = np.zeros((self.max_context_size, 6))  # (52,6)
        box_info[:len(context), 0] = [o.get_bbox().cx for o in context]
        box_info[:len(context), 1] = [o.get_bbox().cy for o in context]
        box_info[:len(context), 2] = [o.get_bbox().cz for o in context]
        box_info[:len(context), 3] = [o.get_bbox().lx for o in context]
        box_info[:len(context), 4] = [o.get_bbox().ly for o in context]
        box_info[:len(context), 5] = [o.get_bbox().lz for o in context]
        box_corners = np.zeros((self.max_context_size, 8, 3))
        box_corners[:len(context)] = [o.get_bbox().corners for o in context]
        if self.object_transformation is not None:
            samples = self.object_transformation(samples)

        res['scan_id'] = scan_id
        res['context_size'] = len(samples)
        res['objects'] = pad_samples(samples, self.max_context_size)

        res['class_labels'] = semantic_labels_of_context(context, self.max_context_size, scan, self.class_to_idx)
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool_)
        target_class_mask[:len(context)] = [target.semantic_label(scan) == o.semantic_label(scan) for o in context]
        res['target_class'] = self.class_to_idx[target.semantic_label(scan)]

        res['target_pos'] = target_pos
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['box_info'] = box_info
        res['box_corners'] = box_corners

        if self.visualization:
            object_ids = np.zeros((self.max_context_size))
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = self.references.loc[index]['utterance']
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id

        return res


def make_data_loaders(args, referit_data, class_to_idx, scans, mean_rgb):
    data_loaders = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'val']

    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb)

    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = args.max_distractors if split == 'train' else args.max_test_objects - 1
        # if split == test remove the utterances of unique targets
        if split == 'val':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            logging.info("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            logging.info("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            logging.info("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

        dataset = Nr3dDataset(references=d_set,
                              scans=scans,
                              points_per_object=args.points_per_object,
                              max_distractors=max_distractors,
                              class_to_idx=class_to_idx,
                              object_transformation=object_transformation,
                              mapping=args.language_word_label)

        seed = None
        if split == 'val':
            seed = args.random_seed

        if split == 'train' and len(dataset) % args.batch_size == 1:
            print('dropping last batch during training')
            drop_last = True
        else:
            drop_last = False
        shuffle = split == 'train'

        worker_init_fn = lambda x: np.random.seed(seed)

        data_loaders[split] = DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.n_workers,
                                         shuffle=shuffle,
                                         drop_last=drop_last,
                                         pin_memory=False,
                                         worker_init_fn=worker_init_fn)
    return data_loaders


def make_test_data_loader(args, referit_data, class_to_idx, scans, mean_rgb):
    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb)
    max_distractors = args.max_test_objects - 1

    def multiple_targets_utterance(x):
        _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
        return len(distractors_ids) > 0

    multiple_targets_mask = referit_data.apply(multiple_targets_utterance, axis=1)
    d_set = referit_data[multiple_targets_mask]
    d_set.reset_index(drop=True, inplace=True)
    print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
    print("removed {} utterances from the test set that don't have multiple distractors".format(
        np.sum(~multiple_targets_mask)))
    print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

    dataset = Nr3dDataset(references=referit_data,
                          scans=scans,
                          points_per_object=args.points_per_object,
                          max_distractors=max_distractors,
                          class_to_idx=class_to_idx,
                          object_transformation=object_transformation,
                          visualization=True,
                          mapping=args.language_word_label)
    seed = args.random_seed

    worker_init_fn = lambda x: np.random.seed(seed)

    data_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             num_workers=args.n_workers,
                             shuffle=False,
                             drop_last=False,
                             pin_memory=False,
                             worker_init_fn=worker_init_fn)
    return data_loader
