"""
TODO: add description

The MIT License (MIT)
Originally created at 7/13/20, for Python 3.x
Copyright (c) 2020 Panos Achlioptas (pachlioptas@gmail.com) & Stanford Geometric Computing Lab
"""

import pandas as pd

from .utterances import is_explicitly_view_dependent
from dataset.utils import decode_stimulus_string
from torch.utils.data import DataLoader
from tools.test_api import detailed_predictions_on_dataset
import numpy as np


def analyze_predictions(model, dataset, device, args, out_file=None, tokenizer=None):
    """
    :param dataset:
    :param net_stats:
    :param pad_idx:
    :return:
    # TODO Panos Post 17 July : clear
    """

    references = dataset.references

    # # YOU CAN USE Those to VISUALIZE PREDICTIONS OF A SYSTEM.
    # confidences_probs = stats['confidences_probs']  # for every object of a scan it's chance to be predicted.
    # objects = stats['contrasted_objects'] # the object-classes (as ints) of the objects corresponding to the confidences_probs
    # context_size = (objects != pad_idx).sum(1) # TODO-Panos assert same as from batch!
    # target_ids = references.instance_type.apply(lambda x: class_to_idx[x])

    hardness = references.stimulus_id.apply(lambda x: decode_stimulus_string(x)[2])
    view_dep_mask = is_explicitly_view_dependent(references)
    easy_context_mask = hardness <= 2

    test_seeds = [args.random_seed, 1, 10, 20, 100]
    net_stats_all_seed = []
    for seed in test_seeds:
        worker_init_fn = lambda x: np.random.seed(seed)
        d_loader = DataLoader(dataset,
                              batch_size=args.batch_size,
                              num_workers=args.n_workers,
                              shuffle=False,
                              drop_last=False,
                              pin_memory=False,
                              worker_init_fn=worker_init_fn)
        assert d_loader.dataset.references is references
        net_stats = detailed_predictions_on_dataset(model, d_loader, args=args, device=device, FOR_VISUALIZATION=True,
                                                    tokenizer=tokenizer)
        net_stats_all_seed.append(net_stats)
    # if visualize_output:
    #     from referit3d.utils import pickle_data
    #     pickle_data(out_file[:-4] + 'all_vis.pkl', net_stats_all_seed)

    all_accuracy = []
    view_dep_acc = []
    view_indep_acc = []
    easy_acc = []
    hard_acc = []
    among_true_acc = []

    for stats in net_stats_all_seed:
        got_it_right = stats['guessed_correctly']
        all_accuracy.append(got_it_right.mean() * 100)
        view_dep_acc.append(got_it_right[view_dep_mask].mean() * 100)
        view_indep_acc.append(got_it_right[~view_dep_mask].mean() * 100)
        easy_acc.append(got_it_right[easy_context_mask].mean() * 100)
        hard_acc.append(got_it_right[~easy_context_mask].mean() * 100)

        got_it_right = stats['guessed_correctly_among_true_class']
        among_true_acc.append(got_it_right.mean() * 100)

    acc_df = pd.DataFrame({'hard': hard_acc, 'easy': easy_acc,
                           'v-dep': view_dep_acc, 'v-indep': view_indep_acc,
                           'all': all_accuracy, 'among-true': among_true_acc})
    print(acc_df)
    acc_df.to_csv(out_file[:-4] + '.csv', index=False)

    pd.options.display.float_format = "{:,.1f}".format
    descriptive = acc_df.describe().loc[["mean", "std"]].T

    if out_file is not None:
        with open(out_file, 'w') as f_out:
            f_out.write(descriptive.to_latex())
    return descriptive

