import os
import numpy as np
import logging
import torchvision.datasets as datasets


class RGBToGray:
    def __call__(self, img):
        return img.convert("L")


def flatten_image(img):
    """Function to flatten the image"""
    return img.view(-1)


def write_file(filepath, filename, data):
    """Function to write contents into the current file
    Args:
        filepath: path of the file
        filename: name of the file
        data: contents to be written into the file
    """
    if not os.path.isdir(filepath):
        os.makedirs(filepath)
    with open(os.path.join(filepath, filename), 'a') as f:
        f.write('\n{}'.format(data))


def save_data(base_dir, filename, data):
    """Function to save data in specific file path
    Args:
        base_dir: directory of the file
        filename: name of the file
        data: data to be saved
    """
    if os.path.isdir(base_dir) == False:
        os.makedirs(base_dir)
    np.save(os.path.join(base_dir, filename), data)


def np_load(path):
    """Function to load data"""
    loaded = np.load(path, allow_pickle=True)
    return loaded


def set_logger(logger_path, name):
    path = os.path.join(logger_path, '{}.log'.format(name))
    if not os.path.exists(path) and not os.path.exists(logger_path):
        os.makedirs(logger_path)
        with open(path, 'w'):
            # create an empty logger file
            pass
        
    if name == 'server':
        logger = logging.getLogger('Asynchronous Parallel Continual Learning server')
    else:
        logger = logging.getLogger('Asynchronous Parallel Continual Learning {}'.format(name))
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    return logger


def compute_target_idx(raw_data_path, dataset, task_dataset):
    """Return the list of indices of the needed output for the specific task.
    Args:
        raw_data_path: path of the raw data
        dataset: the name of the whole dataset
        task_dataset: dataset for the task
    """
    if dataset == 'EMNIST':
        original_dataset = datasets.ImageFolder(os.path.join(raw_data_path, dataset) + '/train')
    elif dataset == 'CIFAR100':
        original_dataset = datasets.CIFAR100(os.path.join(raw_data_path, dataset))
    elif dataset == 'Tiny-ImageNet':
        original_dataset = datasets.ImageFolder(os.path.join(raw_data_path, dataset) + '/tiny-imagenet-200/train')

    if dataset in ['EMNIST', 'Tiny-ImageNet']: 
        targets = []
        for key in original_dataset.class_to_idx.keys():
            targets.append(key)

        results = []

        for key in task_dataset.class_to_idx.keys():
            result = targets.index(key)
            results.append(result)
        return results, targets

    elif dataset == 'CIFAR100':
        data, label = task_dataset[0], task_dataset[1]
        result = np.unique(label)
        return result


def convert_label_back(task_dataset, original_classes, label):
    """Function to convert the labels in each iteration of dataloader into its class index in original dataset.
    Args:
        task_dataset; the dataset for the task
        original_classes: original classes
        label: the label for the current iteration of the data loader
    """
    idx_to_class = {value: key for key, value in task_dataset.class_to_idx.items()}
    for idx, val in enumerate(label):
        label[idx] = original_classes.index(idx_to_class[val.item()])
    return label

