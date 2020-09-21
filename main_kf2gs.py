import os
import datetime

from datasets.hdg import HdgDataset
from models.KeypointFeaturesToGestureStatic import KeypointFeaturesToGestureStatic
from utils.metrics import AverageMeter
from utils.data import save_checkpoint, add_pr_curve_tensorboard

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter


def main():
    # Use GPU if available
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    if use_gpu:
        gpu = torch.cuda.current_device()
        print(f"Currently using GPU {gpu}")
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(40)
    else:
        print("Currently using CPU")

    # Load a dataset of keypoints
    dataset = HdgDataset()
    total_size = len(dataset)
    print(f'Total dataset size {total_size}')

    # Sample the dataset into training and validation sets by ratio

    train_pct = 0.85
    validate_pct = 0.10
    test_pct = 0.05
    assert train_pct + validate_pct + test_pct == 1.0

    train_size = int(train_pct * total_size)
    validate_size = int(validate_pct * total_size)
    test_size = total_size - train_size - validate_size
    dataset_lengths = [train_size, validate_size, test_size]
    print(f'Train %: {train_pct}, Train samples: {train_size}')
    print(f'Validation %: {validate_pct}, Validation samples: {validate_size}')
    print(f'Test %: {test_pct}, Test samples: {test_size}')

    batch_size = 16
    num_epochs = 100
    # num_iter = math.ceil(train_size / batch_size)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.dataset.random_split(dataset, dataset_lengths,
                                                                                       generator=torch.Generator().manual_seed(
                                                                                           41))

    pin_memory = True if use_gpu else False
    # Create the DataLoaders
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   # num_workers=4,
                                                   pin_memory=pin_memory)

    validate_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=True,
                                                      # num_workers=4,
                                                      pin_memory=pin_memory)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  # num_workers=4,
                                                  pin_memory=pin_memory)

    # Create the model
    model = KeypointFeaturesToGestureStatic().to(device)
    print(f'Model {model}')

    # Add model graph to Tensorboard
    tensorboard_root = '/home/thwang/runs'
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
    tensorboard_save_path = os.path.join(tensorboard_root, 'keypt_features_to_gesture_static', timestamp)
    writer = SummaryWriter(tensorboard_save_path)
    dataiter = iter(validate_dataloader)
    keypt_volume, angles, normal, _ = dataiter.next()
    keypt_volume = keypt_volume.to(device)
    angles = angles.to(device)
    normal = normal.to(device)
    writer.add_graph(model, input_to_model=(keypt_volume, angles, normal))

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=0,
                                 amsgrad=False
                                 )

    best_loss = 100000

    for epoch in range(num_epochs):
        train_loss = train(dataloader=train_dataloader,
                           model=model,
                           optimizer=optimizer,
                           use_gpu=use_gpu,
                           device=device,
                           curr_epoch=epoch, )

        validate_loss = validate(dataloader=validate_dataloader,
                                 model=model,
                                 use_gpu=use_gpu,
                                 device=device,
                                 curr_epoch=epoch, )

        test_loss = test(dataset=dataset,
                         dataloader=test_dataloader,
                         model=model,
                         writer=writer,
                         use_gpu=use_gpu,
                         device=device,
                         pr_curve=False, )

        loss_dict = {
            'train': train_loss.avg,
            'validation': validate_loss.avg,
            'test': test_loss.avg
        }

        writer.add_scalars('loss', loss_dict, epoch)

        is_best = validate_loss.avg < best_loss
        best_loss = min(validate_loss.avg, best_loss)
        checkpoint_dict = {
            'epoch': epoch + 1,
            'arch': KeypointFeaturesToGestureStatic,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss,
        }
        filename= f'checkpoints/kf2gs.pt'
        save_checkpoint(state=checkpoint_dict,
            is_best=is_best,
            filename=filename,)

        print(f'Epoch {epoch + 1}: Losses: {loss_dict}')

    # PR curves in Tensorboard for the best model
    best_model = KeypointFeaturesToGestureStatic()
    best_model.load_model_for_inference('checkpoints/kf2gs_model_only.pt')
    if use_gpu:
        best_model.to(device)
    test(dataset=dataset,
         dataloader=test_dataloader,
         model=best_model,
         writer=writer,
         use_gpu=use_gpu,
         device=device,
         pr_curve=True,)
    writer.close()
    return


def train(dataloader, model, optimizer, use_gpu, device, curr_epoch):
    model.train()
    avg_loss = AverageMeter()
    for i, (keypt_volume, angles, normal, targets) in enumerate(dataloader):
        keypt_volume = torch.autograd.Variable(keypt_volume)
        angles = torch.autograd.Variable(angles)
        normal = torch.autograd.Variable(normal)
        targets = torch.autograd.Variable(targets)
        if use_gpu:
            keypt_volume = keypt_volume.to(device)
            angles = angles.to(device)
            normal = normal.to(device)
            targets = targets.to(device)

        optimizer.zero_grad()

        pred = model(keypt_volume, angles, normal)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred, targets)
        if use_gpu:
            avg_loss.update(loss.cpu())
        else:
            avg_loss.update(loss)
        loss.backward()
        optimizer.step()

    # avg loss in the epoch
    return avg_loss


def validate(dataloader, model, use_gpu, device, curr_epoch):
    model.eval()
    avg_loss = AverageMeter()

    for i, (keypt_volume, angles, normal, targets) in enumerate(dataloader):
        keypt_volume = torch.autograd.Variable(keypt_volume)
        angles = torch.autograd.Variable(angles)
        normal = torch.autograd.Variable(normal)
        targets = torch.autograd.Variable(targets)
        if use_gpu:
            keypt_volume = keypt_volume.to(device)
            angles = angles.to(device)
            normal = normal.to(device)
            targets = targets.to(device)

        with torch.no_grad():
            pred = model(keypt_volume, angles, normal)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred, targets)
        if use_gpu:
            avg_loss.update(loss.cpu())
        else:
            avg_loss.update(loss)

    # avg loss in epoch
    return avg_loss


def test(dataset, dataloader, model, writer, use_gpu, device, pr_curve=False):
    model.eval()
    avg_loss = AverageMeter()
    class_probs = []
    class_preds = []

    with torch.no_grad():
        for i, (keypt_volume, angles, normal, targets) in enumerate(dataloader):
            keypt_volume = torch.autograd.Variable(keypt_volume)
            angles = torch.autograd.Variable(angles)
            normal = torch.autograd.Variable(normal)
            targets = torch.autograd.Variable(targets)
            if use_gpu:
                keypt_volume = keypt_volume.to(device)
                angles = angles.to(device)
                normal = normal.to(device)
                targets = targets.to(device)

            output = model(keypt_volume, angles, normal)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)
            if use_gpu:
                class_probs_batch = [p.to(device) for p in class_probs_batch]
                class_probs.append(class_probs_batch)
                class_preds.append(class_preds_batch.to(device))
            else:
                class_probs.append(class_probs_batch)
                class_preds.append(class_preds_batch)

            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output, targets)
            if use_gpu:
                avg_loss.update(loss.cpu())
            else:
                avg_loss.update(loss)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)
    # plot all the pr curves
    if pr_curve:
        for i in range(len(dataset.gesture_labels)):
            add_pr_curve_tensorboard(writer=writer,
                                     classes=dataset.gesture_labels,
                                     class_index=i,
                                     test_probs=test_probs,
                                     test_preds=test_preds)
    return avg_loss


if __name__ == '__main__':
    main()
