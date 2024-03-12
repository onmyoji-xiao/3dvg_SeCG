import argparse
import os.path

import torch.nn as nn
from data.load_data import *
from tools.utils import *
from transformers import BertTokenizer
from tools.train_api import evaluate_on_dataset

from models.refer_net import ReferIt3DNet_transformer
from dataset.nr3d_dataset import make_test_data_loader
from datetime import datetime
from analysis.deepnet_predictions import analyze_predictions
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description='ReferIt3D Evaluation')

    parser.add_argument('-scannet-file', type=str, default='../scannet/scannet_00_views.pkl',
                        help='pkl file containing the data of Scannet generated by running ...')
    parser.add_argument('-refer_test_file', type=str, default='./data/referit3d/nr3d_test.csv')
    parser.add_argument('--weight', type=str, default='./checkpoints/ckpt_nr3d.pth')

    parser.add_argument('--n-workers', type=int, default=0, help='number of data loading workers')

    parser.add_argument('--max-distractors', type=int, default=51,
                        help='Maximum number of distracting objects to be drawn from a scan.')
    parser.add_argument('--max-test-objects', type=int, default=88)
    parser.add_argument('--points-per-object', type=int, default=1024,
                        help='points sampled to make a point-cloud per object of a scan.')
    parser.add_argument('--random-seed', type=int, default=2020,
                        help='Control pseudo-randomness (net-wise, point-cloud sampling etc.) fostering reproducibility.')

    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--n-gpus', type=int, default=1, help='number gpu devices. [default: 1]')
    parser.add_argument('--batch-size', type=int, default=12, help='batch size per gpu. [default: 32]')

    parser.add_argument('--bert-pretrain-path', type=str, default='../pretrained/bert')
    parser.add_argument('--sem_encode', type=bool, default=True)
    # MVT
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--gat', type=bool, default=True)
    parser.add_argument('--clf-semantic', type=bool, default=True)
    parser.add_argument('--lay-number', type=int, default=2)
    parser.add_argument('--multi_pos', type=bool, default=True)

    parser.add_argument('--view-number', type=int, default=4)
    parser.add_argument('--rotate-number', type=int, default=4)
    parser.add_argument('--aggregate-type', type=str, default='avg')
    parser.add_argument('--encoder-layer-num', type=int, default=3)
    parser.add_argument('--decoder-layer-num', type=int, default=4)
    parser.add_argument('--decoder-nhead-num', type=int, default=8)
    parser.add_argument('--object-latent-dim', type=int, default=768)
    parser.add_argument('--inner-dim', type=int, default=768)
    parser.add_argument('--dropout-rate', type=float, default=0.15)
    parser.add_argument('--lang-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing the target via '
                                                                          'language only is added.')
    parser.add_argument('--obj-cls-alpha', type=float, default=0.5, help='if > 0 a loss for guessing for each segmented'
                                                                         ' object its class type is added.')

    args = parser.parse_args()
    return args


def load_test_data(pkl_scannet_file, refer_test, clf_s):
    pkls = pkl_scannet_file.split(';')
    all_scans = dict()
    for pkl_f in pkls:
        with open(pkl_f, 'rb') as f:
            scans = pickle.load(f)
        scans = {scan.scan_id: scan for scan in scans}
        all_scans.update(scans)
    class_labels = set()
    if clf_s:
        for k, scan in all_scans.items():
            idx = np.array([o.object_id for o in scan.three_d_objects])
            class_labels.update([o.semantic_label(scan) for o in scan.three_d_objects])
            assert np.all(idx == np.arange(len(idx)))
    else:
        for k, scan in all_scans.items():
            idx = np.array([o.object_id for o in scan.three_d_objects])
            class_labels.update([o.instance_label for o in scan.three_d_objects])
            assert np.all(idx == np.arange(len(idx)))

    class_to_idx = {}
    i = 0
    for el in sorted(class_labels):
        class_to_idx[el] = i
        i += 1
    class_to_idx['pad'] = len(class_to_idx)

    referit_data = pd.read_csv(refer_test)
    referit_data = referit_data[['tokens', 'instance_type', 'scan_id', 'is_train',
                                 'dataset', 'target_id', 'utterance', 'stimulus_id']]
    referit_data.tokens = referit_data['tokens'].apply(literal_eval)

    return all_scans, referit_data, class_to_idx


if __name__ == '__main__':
    # Parse arguments
    args = parse_arguments()
    tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    test_scans, referit_data, class_to_idx = load_test_data(args.scannet_file, args.refer_test_file, args.clf_semantic)

    test_scans = trim_scans_per_referit3d_data(referit_data, test_scans)

    # mean_rgb
    mean_rgb = [[0.4585, 0.4149, 0.3644]]
    instance2semantic = list(test_scans.values())[0].dataset.instance_cls2semantic_cls
    args.language_word_label = {}
    for k, v in instance2semantic.items():
        args.language_word_label[k] = v
        ss = k.strip().split(' ')
        if len(ss) > 1 and ss[-1] not in instance2semantic:
            args.language_word_label[ss[-1]] = v
    data_loader = make_test_data_loader(args, referit_data, class_to_idx, test_scans, mean_rgb)

    # Prepare GPU environment
    if torch.cuda.is_available():
        device = torch.device('cuda')
        seed_training_code(args.random_seed)
    else:
        device = torch.device('cpu')

    class_name_list = []
    for cate in class_to_idx:
        class_name_list.append(cate)

    n_classes = len(class_to_idx) - 1  # -1 to ignore the <pad> class
    pad_idx = class_to_idx['pad']
    model = ReferIt3DNet_transformer(args, n_classes,  ignore_index=pad_idx)

    checkpoint = torch.load(args.weight)
    model.load_state_dict(checkpoint['model'], strict=True)
    print('loaded model weight of ' + args.weight)

    model = model.to(device)
    model.eval()
    out_file = os.path.dirname(args.weight) + '/res.csv'
    res = analyze_predictions(model, data_loader.dataset, device, args, out_file=out_file, tokenizer=tokenizer)
    print(res)
