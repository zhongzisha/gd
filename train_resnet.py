import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数
from torch.utils.tensorboard import SummaryWriter
import cv2
import shutil


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(images, labels):
    # if one_channel:
    #     img = img.mean(dim=0)
    # img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    # if one_channel:
    #     plt.imshow(npimg, cmap="Greys")
    # else:
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    inp = images.clone().numpy().transpose((0, 2, 3, 1))
    h, w = inp.shape[1:3]
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1) * 255
    inp = inp.astype(np.uint8)
    newinps = []
    for image, label in zip(inp, labels):
        image = np.ascontiguousarray(image)
        cv2.putText(image, text=str(label), org=(w // 2, h // 2),
                    fontFace=1, fontScale=1,
                    color=(0, 0, 255) if label == 0 else (0, 255, 255))
        newinps.append(image)
    return torch.from_numpy(np.stack(newinps).transpose((0, 3, 1, 2)))


def train_model(model, criterion, optimizer, scheduler, num_epochs=25,
                save_root=''):
    since = time.time()

    tb_dir = '%s/tb' % save_root
    if os.path.exists(tb_dir):
        shutil.rmtree(tb_dir, ignore_errors=True)
    writer = SummaryWriter(tb_dir)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            all_labels = []
            all_preds = []
            running_loss = 0.0
            for step, (inputs, labels) in enumerate(dataloaders[phase]):

                if step % 100 == 0:
                    # show images
                    img_grid = matplotlib_imshow(inputs, labels)
                    # create grid of images
                    img_grid = torchvision.utils.make_grid(img_grid)
                    # write to tensorboard
                    writer.add_image('%s/images' % phase, img_grid, global_step=epoch * len(dataloaders[phase]) + step)

                inputs = inputs.to(device)
                labels = labels.to(device)
                all_labels.append(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    all_preds.append(preds)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            all_labels = torch.cat(all_labels).cpu().numpy()
            all_preds = torch.cat(all_preds).cpu().numpy()
            cm = confusion_matrix(all_labels, all_preds)
            print(phase, cm)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), '%s/best.pt' % save_root)

            # ...log the running loss
            writer.add_scalar('%s_loss' % phase,
                              epoch_loss,
                              epoch * len(dataloaders[phase]))

        print()

        # import pdb
        # pdb.set_trace()
        torch.save(model.state_dict(), '%s/epoch-%d.pt' % (save_root, epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#

def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--train_subset", type=str, default='train')
    parser.add_argument("--val_subset", type=str, default='val')
    parser.add_argument("--save_root", type=str, default='')
    parser.add_argument("--netname", type=str, default='resnet50')
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100)

    return parser.parse_args()


"""
python train_resnet.py ^
--data_dir E:\ganta_patch_classification ^
--train_subset train1 ^
--val_subset val1 ^
--save_root E:\ganta_patch_cls_results ^
--netname resnet50 ^
--batchsize 8 ^
--lr 0.001 ^
--num_epochs 100
"""

def train(args):

    data_dir = args.data_dir
    netname = args.netname
    batchsize = args.batchsize
    lr = args.lr
    num_epochs = args.num_epochs

    save_root = '%s/%s/bs%d_lr%f_epochs%d' % (args.save_root, netname, batchsize, lr, num_epochs)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    data_transforms = {
        'train': transforms.Compose([
            # transforms.RandomRotation(180),
            transforms.GaussianBlur(3, (0.01, 0.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.05, saturation=0.05, hue=0.05),
            transforms.RandomAffine(degrees=180, translate=(0.01, 0.03), scale=(0.9, 1.1),
                                    shear=(1, 2)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            # transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {'train': datasets.ImageFolder(os.path.join(data_dir, args.train_subset), data_transforms['train']),
                      'val': datasets.ImageFolder(os.path.join(data_dir, args.val_subset), data_transforms['val'])}

    # weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, len(image_datasets['train'].classes))
    # weights = torch.DoubleTensor(weights)
    # print('weights', len(weights))
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    # dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], num_workers=2, batch_size=batchsize,
    #                                                     sampler=sampler),
    #                'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batchsize,
    #                                                   shuffle=False, num_workers=2)}
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], num_workers=2, batch_size=batchsize,
                                                        shuffle=True),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batchsize,
                                                      shuffle=False, num_workers=2)}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = None
    if netname == 'resnet18':
        model_ft = models.resnet18(pretrained=True)
    elif netname == 'resnet50':
        model_ft = models.resnet50(pretrained=True)
    elif netname == 'resnet101':
        model_ft = models.resnet101(pretrained=True)

    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    # criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8], device=device))
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[70, 90], gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=num_epochs, save_root=save_root)

    visualize_model(model_ft)

    plt.show()

    import pdb

    pdb.set_trace()

