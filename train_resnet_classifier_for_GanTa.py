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
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import shutil

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

import argparse
import cv2


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



def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

    def __len__(self) -> int:
        return len(self.samples)



class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default='train')    # train or test
    parser.add_argument("--data_dir", type=str, default='')
    parser.add_argument("--train_subset", type=str, default='train')
    parser.add_argument("--val_subset", type=str, default='val')
    parser.add_argument("--test_subset", type=str, default='val')
    parser.add_argument("--save_root", type=str, default='')
    parser.add_argument("--netname", type=str, default='resnet50')
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=100)

    return parser.parse_args()


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


def test(args):
    data_dir = args.data_dir
    netname = args.netname
    batchsize = args.batchsize
    lr = args.lr
    num_epochs = args.num_epochs
    subset_prefix = args.test_subset

    save_root = '%s/%s/bs%d_lr%f_epochs%d' % (args.save_root, netname, batchsize, lr, num_epochs)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
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

    # dataset = ImageFolder(os.path.join(data_dir, 'val1'),  data_transforms['val'])
    dataset = ImageFolder(os.path.join(data_dir, subset_prefix), data_transforms['val'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False, num_workers=0)
    class_names = dataset.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    modelc = None
    if netname == 'resnet18':
        modelc = models.resnet18(pretrained=True)
    elif netname == 'resnet50':
        modelc = models.resnet50(pretrained=True)
    elif netname == 'resnet101':
        modelc = models.resnet101(pretrained=True)

    num_ftrs = modelc.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    modelc.fc = nn.Linear(num_ftrs, 2)
    modelc.load_state_dict(torch.load('%s/best.pt' % save_root, map_location=device))
    modelc = modelc.to(device)
    modelc.eval()
    prob_op = nn.Softmax(dim=1)

    all_labels = []
    all_probs = []
    all_preds = []
    all_paths = []
    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = modelc(inputs)
            probs = prob_op(outputs)
            _, preds = torch.max(outputs, 1)

            # import pdb
            # pdb.set_trace()

            all_labels.append(labels)
            all_probs.append(probs)
            all_preds.append(preds)
            all_paths += paths

    all_labels = torch.cat(all_labels).cpu().numpy()
    all_probs = torch.cat(all_probs).cpu().numpy()
    all_preds = torch.cat(all_preds).cpu().numpy()

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)

    save_roots = ['%s/%s/neg_to_pos' % (save_root, subset_prefix),
                  '%s/%s/pos_to_neg' % (save_root, subset_prefix)]
    for p in save_roots:
        if os.path.exists(p):
            shutil.rmtree(p, ignore_errors=True)
            os.makedirs(p)
        else:
            os.makedirs(p)
    for i, (label, pred, prob, path) in enumerate(zip(all_labels, all_preds, all_probs, all_paths)):
        if label == 0 and pred == 1:
            save_filename = os.path.join(save_roots[0], path.split(os.sep)[-1].replace('.jpg', '_%f.jpg' % (prob[1])))
            shutil.copy(path, save_filename)
        elif label == 1 and pred == 0:
            save_filename = os.path.join(save_roots[1], path.split(os.sep)[-1].replace('.jpg', '_%f.jpg' % (prob[1])))
            shutil.copy(path, save_filename)

    # import pdb
    # pdb.set_trace()


"""
python train_resnet.py ^
--action train ^
--data_dir E:\ganta_patch_classification ^
--train_subset train1 ^
--val_subset val1 ^
--save_root E:\ganta_patch_cls_results ^
--netname resnet50 ^
--batchsize 8 ^
--lr 0.001 ^
--num_epochs 100

python train_resnet.py ^
--action test ^
--data_dir E:\ganta_patch_classification ^
--train_subset train1 ^
--val_subset val1 ^
--test_subset val1 ^
--save_root E:\ganta_patch_cls_results ^
--netname resnet50 ^
--batchsize 8 ^
--lr 0.001 ^
--num_epochs 100
"""

if __name__ == '__main__':
    args = get_args()
    if args.action == 'train':
        train(args)
    elif args.action == 'test':
        test(args)
