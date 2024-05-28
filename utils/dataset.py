import torch
import torchvision.datasets as datasets
from PIL import Image
import os
import numpy as np
import subprocess
import bitstring
import torchvision.transforms as transforms
from tqdm import tqdm
import glob 
from shutil import move
from utils.misc import flatten_image, RGBToGray


emnist_transform = transforms.Compose([
    RGBToGray(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.137, ), std=(0.3081,)),
    transforms.Lambda(flatten_image)
])

cifar100_transform = transforms.Compose([
    RGBToGray(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[x/255 for x in [125.3, 123.0, 113.9]], std=[x/255 for x in [63.0, 62.1, 66.7]]),
    transforms.Lambda(flatten_image)
])

imagenet_transform = transforms.Compose([
    RGBToGray(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.448, 0.397], std=[0.272, 0.265, 0.274]),
    transforms.Lambda(flatten_image)
])

class DataGenerator:
    """Class for generating data"""
    def __init__(self, args, logger):
        self.args = args
        self.task_cnt = -1
        self.emnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])

        self.cifar100_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255 for x in [125.3, 123.0, 113.9]], std=[x/255 for x in [63.0, 62.1, 66.7]])
        ])

        self.imagenet_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.448, 0.397], std=[0.272, 0.265, 0.274])
        ])

    def get_raw_data(self):
        """Function to download needed datasets"""
        script_path = 'scripts/get_raw.sh'
        # subprocess.run(['sh', script_path], shell=False)

        # CIFAR100
        # cifar100 = datasets.CIFAR100(root=self.args.raw_data_path + 'CIFAR100/', train=True, download=True)

        # EMNIST
        # self.prepare_emnist()

        # Tiny-ImageNet
        self.prepare_imagenet()

    def prepare_emnist(self):
        """Function to prepare EMNIST dataset"""
        for name in ['train', 'test']:
            images_file = 'data/raw/EMNIST/gzip/emnist-byclass-{}-images-idx3-ubyte'.format(name)
            labels_file = 'data/raw/EMNIST/gzip/emnist-byclass-{}-labels-idx1-ubyte'.format(name)
            map_file = 'data/raw/EMNIST/gzip/emnist-byclass-mapping.txt'
            output_dir = 'data/raw/EMNIST/{}'.format(name)

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # create class-char map
            with open(map_file, 'r') as f: 
                lines = f.readlines()
                labelmap = {}
                for line in lines:
                    class_id = int(line.split(' ')[0])
                    char_id = int(line.split(' ')[1])
                    labelmap[class_id] = char_id

            # read images binfile header
            images_bitstream = bitstring.ConstBitStream(filename=images_file)
            images_bitstream.read('int:32')
            n_images = images_bitstream.read('int:32')
            img_width = images_bitstream.read('int:32')
            img_height = images_bitstream.read('int:32')

            # read labels binfile header
            labels_bitstream = bitstring.ConstBitStream(filename=labels_file)
            labels_bitstream.read('int:32')
            n_labels = labels_bitstream.read('int:32')

            # validation
            assert n_images == n_labels, 'the number of images is not the same as that of images.'
            n_samples = n_images

            cnt = 0
            for i in range(n_samples):
                cnt += 1

                # read a single label record
                record_label = labels_bitstream.read('uint:8')

                # reconstruct the label id
                label = np.uint8(record_label)

                # decoded label character
                character = labelmap[label]

                # create subdirectory (if necessary)
                subdir = os.path.join(output_dir, str(character))
                if os.path.exists(subdir) == False:
                    os.makedirs(subdir)

                # read a single image record
                record_image = images_bitstream.readlist('%d*uint:8' % (img_width * img_height))

                # reconstruct the image data
                pixel_data = np.array(record_image, dtype=np.uint8).reshape(img_height, img_width)
                pixel_data = pixel_data.T
                image = Image.fromarray(pixel_data)

                # save image
                fname = os.path.join(subdir, '{}.png'.format(cnt))
                image.save(fname)
            
    def prepare_imagenet(self):
        """Function to prepare imagenet dataset"""
        val_dict = dict() 
        target_folder = os.path.join(self.args.raw_data_path, '{}/tiny-imagenet-200/val/'.format(self.args.dataset))
        with open(os.path.join(self.args.raw_data_path, '{}/tiny-imagenet-200/val/val_annotations.txt'.format(self.args.dataset))) as f:
            for line in f.readlines():
                split_line = line.split('\t')
                val_dict[split_line[0]] = split_line[1]
        
        paths = glob.glob(os.path.join(self.args.raw_data_path, self.args.dataset) + '/tiny-imagenet-200/val/images/*')
        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            if not os.path.exists(target_folder + str(folder)):
                os.makedirs(target_folder + str(folder) + '/images')

        for path in paths:
            file = path.split('/')[-1]
            folder = val_dict[file]
            dest = target_folder + str(folder) + '/images/' + str(file)
            move(path, dest)

        os.rmdir(os.path.join(target_folder, 'images'))

    def generate_tasks(self):
        """Function to generate tasks needed for training"""
        emnist_train = datasets.ImageFolder(root=self.args.raw_data_path + 'EMNIST/train/', transform=self.emnist_transform)
        emnist_test = datasets.ImageFolder(root=self.args.raw_data_path + 'EMNIST/test/', transform=self.emnist_transform)
        cifar100_train = datasets.CIFAR100(root=self.args.raw_data_path + 'CIFAR100/', train=True, download=False, transform=self.cifar100_transform)
        cifar100_test = datasets.CIFAR100(root=self.args.raw_data_path + 'CIFAR100/', train=False, download=False, transform=self.cifar100_transform)
        imagenet_train = datasets.ImageFolder(root=self.args.raw_data_path + 'Tiny-ImageNet/tiny-imagenet-200/train/', transform=self.imagenet_transform)
        imagenet_test = datasets.ImageFolder(root=self.args.raw_data_path + 'Tiny-ImageNet/tiny-imagenet-200/val/', transform=self.imagenet_transform)
        
        for cnt in range(self.args.num_class_sets):
            for name in ['EMNIST', 'CIFAR100', 'Tiny-ImageNet']:
                task_dir = self.args.data_path + name + '_task/{}/'.format(cnt)
                if not os.path.exists(task_dir):
                    os.makedirs(task_dir)
                if name == 'EMNIST':
                    self.emnist_generate_tasks(emnist_train, emnist_test, task_dir)
                # elif name == 'CIFAR100':
                #     self.cifar100_generate_tasks(cifar100_train, cifar100_test, task_dir)
                # elif name == 'Tiny-ImageNet':
                #     self.imagenet_generate_tasks(imagenet_train, imagenet_test, task_dir)

    def emnist_generate_tasks(self, train, test, task_dir):
        """Function to generate tasks for EMNIST, 62 classes into 5 tasks, first three tasks have 12 classes"""
        classes = np.array([int(cla) for cla in train.classes])
        np.random.shuffle(classes)
        
        tasks = []

        # first three tasks have 12 classes, 
        start = 0
        for cut in [12, 12, 12, 13, 13]:
            end = start + cut
            tasks.append(classes[start: end])
            start = end

        for idx, task in enumerate(tasks):
            task_path = task_dir + '/task_{}'.format(idx)
            train_path = task_path + '/train'
            test_path = task_path + '/test'
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            if not os.path.exists(test_path):
                os.makedirs(test_path)

            for item in task:
                train_source_path = os.path.join(self.args.raw_data_path + 'EMNIST/train/', str(item))
                train_dest_path = train_path
                test_source_path = os.path.join(self.args.raw_data_path + 'EMNIST/test/', str(item))
                test_dest_path = test_path
                subprocess.run(['cp', '-r', train_source_path, train_dest_path])
                subprocess.run(['cp', '-r', test_source_path, test_dest_path])

    def cifar100_generate_tasks(self, train, test, task_dir):
        """Function to genreate tasks for CIFAR100"""
        num_classes = len(train.classes)
        label_to_data = {i: {'train': [], 'test': []} for i in range(num_classes)}

        for _, (data, label) in tqdm(enumerate(train)):
            label_to_data[label]['train'].append(data)

        for _, (data, label) in tqdm(enumerate(test)):
            label_to_data[label]['test'].append(data)

        num_per_train = len((label_to_data)[0]['train'])
        num_per_test = len(label_to_data[0]['test'])

        idx = np.random.permutation(np.arange(num_classes))

        start = 0
        for task_num, cut in tqdm(enumerate([10] * 10)):
            end = start + cut
            x_tr = torch.cat([torch.tensor(np.array(label_to_data[item]['train'])) for item in idx[start: end]], dim=0)
            y_tr = torch.cat([torch.tensor(np.array([item] * num_per_train)) for item in idx[start: end]], dim=0)
            x_te = torch.cat([torch.tensor(np.array(label_to_data[item]['test'])) for item in idx[start: end]], dim=0)
            y_te = torch.cat([torch.tensor(np.array([item] * num_per_test)) for item in idx[start: end]], dim=0)
            task_path = task_dir + '/task_{}'.format(task_num)
            if not os.path.exists(task_path):
                os.makedirs(task_path)
            torch.save((x_tr, y_tr), os.path.join(task_path, 'train.pt'))
            torch.save((x_te, y_te), os.path.join(task_path, 'test.pt'))
            start = end

    def imagenet_generate_tasks(self, train, test, task_dir):
        """Function to generate tasks for Tiny-ImageNet"""
        classes = np.array([idx for idx, cla in enumerate(train.classes)])
        class_map = {idx: cla for idx, cla in enumerate(train.classes)}
        np.random.shuffle(classes)
        
        tasks = []

        start = 0
        for _, cut in enumerate([20] * 10):
            end = start + cut
            tasks.append(classes[start: end])
            start = end
        
        for idx, task in enumerate(tasks):
            task_path = task_dir + 'task_{}'.format(idx)
            train_path = task_path + '/train'
            test_path = task_path + '/test'
            if not os.path.exists(train_path):
                os.makedirs(train_path)
            if not os.path.exists(test_path):
                os.makedirs(test_path)

            for item in task:
                train_source_path = os.path.join(self.args.raw_data_path + 'Tiny-ImageNet/tiny-imagenet-200/train/', str(class_map[item]))
                train_dest_path = train_path
                test_source_path = os.path.join(self.args.raw_data_path + 'Tiny-ImageNet/tiny-imagenet-200/val/', str(class_map[item]))
                test_dest_path = test_path
                subprocess.run(['cp', '-r', train_source_path, train_dest_path])
                subprocess.run(['cp', '-r', test_source_path, test_dest_path])


def prepare_dataset(args, logger):
    """Function to prepare datasets"""
    gen = DataGenerator(args, logger)
    # gen.get_raw_data()
    gen.generate_tasks()
