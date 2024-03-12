import torch
from tools.utils import AverageMeter
import numpy as np
import time


def make_batch_keys(args, extras=None):
    """depending on the args, different data are used by the listener."""
    batch_keys = ['objects', 'tokens', 'target_pos']  # all models use these
    if extras is not None:
        batch_keys += extras

    if args.obj_cls_alpha > 0:
        batch_keys.append('class_labels')

    if args.lang_cls_alpha > 0:
        batch_keys.append('target_class')

    return batch_keys


def single_epoch_train(model, data_loader, optimizer, device, pad_idx, args, tokenizer, epoch, logger):
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()

    # Set the model in training mode
    model.train()
    np.random.seed()  # call this to change the sampling of the point-clouds
    batch_keys = make_batch_keys(args)
    for i, batch in enumerate(data_loader):
        st = time.time()
        # Move data to gpu
        for k in batch_keys:
            if k in batch:
                if isinstance(batch[k], list):
                    continue
                batch[k] = batch[k].to(device)

        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].to(device)
        batch['lang_tokens'] = lang_tokens

        # Forward pass
        LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = model(batch)
        LOSS = LOSS.mean()

        res = {}
        res['logits'] = LOGITS
        res['class_logits'] = CLASS_LOGITS
        res['lang_logits'] = LANG_LOGITS
        # Backward
        optimizer.zero_grad()
        LOSS.backward()
        optimizer.step()

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(LOSS.item(), batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)

        if (i + 1) % args.log_iter == 0:
            logger.info('epoch %d batch %d/%d total_loss: %f time: %f s/batch' % (
                epoch, i + 1, len(data_loader), LOSS.item(), time.time() - st))
    metrics['train_total_loss'] = total_loss_mtr.avg
    metrics['train_referential_acc'] = ref_acc_mtr.avg
    metrics['train_object_cls_acc'] = cls_acc_mtr.avg
    metrics['train_txt_cls_acc'] = txt_acc_mtr.avg
    return metrics


@torch.no_grad()
def evaluate_on_dataset(model, data_loader, device, pad_idx, args, randomize=False, tokenizer=None, visual=False):
    # TODO post-deadline, can we replace this func with the train + a 'phase==eval' parameter?
    metrics = dict()  # holding the losses/accuracies
    total_loss_mtr = AverageMeter()
    ref_acc_mtr = AverageMeter()
    cls_acc_mtr = AverageMeter()
    txt_acc_mtr = AverageMeter()
    # Set the model in training mode
    model.eval()

    if randomize:
        np.random.seed()
    else:
        np.random.seed(args.random_seed)

    batch_keys = make_batch_keys(args)

    visual_list = [[], [], [], [], []]
    for i, batch in enumerate(data_loader):
        # Move data to gpu
        for k in batch_keys:
            if isinstance(batch[k], list):
                continue
            batch[k] = batch[k].to(device)

        lang_tokens = tokenizer(batch['tokens'], return_tensors='pt', padding=True)
        for name in lang_tokens.data:
            lang_tokens.data[name] = lang_tokens.data[name].to(device)
        batch['lang_tokens'] = lang_tokens

        # Forward pass
        LOSS, CLASS_LOGITS, LANG_LOGITS, LOGITS = model(batch)
        LOSS = LOSS.mean()
        res = {}
        res['logits'] = LOGITS
        res['class_logits'] = CLASS_LOGITS
        res['lang_logits'] = LANG_LOGITS

        # Update the loss and accuracy meters
        target = batch['target_pos']
        batch_size = target.size(0)  # B x N_Objects
        total_loss_mtr.update(LOSS.item(), batch_size)

        predictions = torch.argmax(res['logits'], dim=1)
        guessed_correctly = torch.mean((predictions == target).double()).item()
        ref_acc_mtr.update(guessed_correctly, batch_size)

        if args.obj_cls_alpha > 0:
            cls_b_acc, _ = cls_pred_stats(res['class_logits'], batch['class_labels'], ignore_label=pad_idx)
            cls_acc_mtr.update(cls_b_acc, batch_size)

        if args.lang_cls_alpha > 0:
            batch_guess = torch.argmax(res['lang_logits'], -1)
            cls_b_acc = torch.mean((batch_guess == batch['target_class']).double())
            txt_acc_mtr.update(cls_b_acc, batch_size)
        if visual:
            visual_list[0].extend(target)
            visual_list[1].extend(predictions)
            visual_list[2].extend(batch['tokens'])
            visual_list[3].extend(batch['scan_id'])
            visual_list[4].extend(batch['object_ids'])
    metrics['test_total_loss'] = total_loss_mtr.avg
    metrics['test_referential_acc'] = ref_acc_mtr.avg
    metrics['test_object_cls_acc'] = cls_acc_mtr.avg
    metrics['test_txt_cls_acc'] = (txt_acc_mtr.avg).cpu()
    if visual:
        return metrics, visual_list
    else:
        return metrics


@torch.no_grad()
def cls_pred_stats(logits, gt_labels, ignore_label):
    """ Get the prediction statistics: accuracy, correctly/wrongly predicted test examples
    :param logits: The output of the model (predictions) of size: B x N_Objects x N_Classes
    :param gt_labels: The ground truth labels of size: B x N_Objects
    :param ignore_label: The label of the padding class (to be ignored)
    :return: The mean accuracy and lists of correct and wrong predictions
    """
    predictions = logits.argmax(dim=-1)  # B x N_Objects x N_Classes --> B x N_Objects
    valid_indices = gt_labels != ignore_label

    predictions = predictions[valid_indices]
    gt_labels = gt_labels[valid_indices]

    correct_guessed = gt_labels == predictions
    assert (type(correct_guessed) == torch.Tensor)

    found_samples = gt_labels[correct_guessed]
    # missed_samples = gt_labels[torch.logical_not(correct_guessed)] # TODO  - why?
    mean_accuracy = torch.mean(correct_guessed.double()).item()
    return mean_accuracy, found_samples
