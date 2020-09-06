import numpy as np
import torch
import shutil
import os


def split2list(images, split, default_split=0.9):
    if isinstance(split, str):
        with open(split) as f:
            split_values = [x.strip() == '1' for x in f.readlines()]
        assert (len(images) == len(split_values))
    elif isinstance(split, float):
        split_values = np.random.uniform(0, 1, len(images)) < split
    else:
        split_values = np.random.uniform(0, 1, len(images)) < default_split
    train_samples = [sample for sample, split in zip(images, split_values) if split]
    test_samples = [sample for sample, split in zip(images, split_values) if not split]
    return train_samples, test_samples


def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)
    print(f'len data {len(data)}')
    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]


def save_checkpoint(state, is_best, filename='checkpoints/k2gs.pt'):
    '''
    Generic model checkpoint save function, but we are using this format:
    {
            'epoch': epoch + 1,
            'arch': KeypointToGestureStatic,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
    }

    Currently saving both just model.state_dict() and the entire JSON separately due
    to buggy PyTorch loading of checkpoints in different project directory structures

    :param state:
    :param is_best:
    :param filename:
    :return:
    '''

    root, ext = os.path.splitext(filename)

    torch.save(state, filename)
    model_only_filename = root + '_model_only' + ext
    torch.save(state['model_state_dict'], model_only_filename)

    if is_best:
        shutil.copyfile(filename, root + '_best' + ext)
        model_only_root, _ = os.path.splitext(model_only_filename)
        shutil.copyfile(model_only_filename, model_only_root + '_best' + ext)

# helper function
def add_pr_curve_tensorboard(writer, classes, class_index, test_probs, test_preds, global_step=0):
    '''
    Takes in a "class_index" from 0 to 9 and plots the corresponding
    precision-recall curve
    '''
    tensorboard_preds = test_preds == class_index
    tensorboard_probs = test_probs[:, class_index]

    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()