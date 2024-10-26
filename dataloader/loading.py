import os
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import pickle

import numpy as np
from torch.utils.data import Dataset
import json


class My_Dataset(Dataset):

    def __init__(self, X, y, transform=None):
        self.images = X.reshape(-1, 1, 52, 52)
        self.one_hot_labels = y
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.transform = transform

        self.classes_dict = {
            "00000000": 0,
            "10000000": 1,
            "01000000": 2,
            "00100000": 3,
            "00010000": 4,
            "00001000": 5,
            "00000100": 6,
            "00000010": 7,
            "00000001": 8,
            "10100000": 9,
            "10010000": 10,
            "10001000": 11,
            "10000010": 12,
            "01100000": 13,
            "01010000": 14,
            "01001000": 15,
            "01000010": 16,
            "00101000": 17,
            "00100010": 18,
            "00011000": 19,
            "00010010": 20,
            "00001010": 21,
            "10101000": 22,
            "10100010": 23,
            "10011000": 24,
            "10010010": 25,
            "10001010": 26,
            "01101000": 27,
            "01100010": 28,
            "01011000": 29,
            "01010010": 30,
            "01001010": 31,
            "00101010": 32,
            "00011010": 33,
            "10101010": 34,
            "10011010": 35,
            "01101010": 36,
            "01011010": 37,
        }

        self.classes_num_dict = {}

        json_str = json.dumps(self.classes_dict, indent=4)
        with open("classes_dict.json", 'w') as json_file:
            json_file.write(json_str)

        self.image_labels = []
        self.images_num = []

        cnt = 0
        for one_hot_label in self.one_hot_labels:
            cnt += 1
            t = ""
            for ch in one_hot_label:
                if ch == None:
                    t += str(0)
                else:
                    t += str(ch)
            label = self.classes_dict[t]
            if label not in self.classes_num_dict:
                self.classes_num_dict[label] = 1
            else:
                self.classes_num_dict[label] += 1
            self.image_labels.append(label)

        for i in sorted(list(self.classes_num_dict.keys())):
            self.images_num.append(self.classes_num_dict[i])

        self.num_classes = len(self.classes_num_dict)

    def __len__(self):
        return sum(self.images_num)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.image_labels[idx]

        if self.transform is not None:
            new_image = np.transpose(image, (1, 2, 0))
            image = self.transform(new_image)
        return image.float(), label


class WAFERMAPDataset(Dataset):
    def __init__(self, data_list, train=True):
        self.trainsize = (64, 64)
        self.train = train
        with open(data_list, "rb", encoding='utf-8') as f:
            tr_dl = pickle.load(f)
        self.data_list = tr_dl

        self.size = len(self.data_list)
        if train:
            self.transform_center = transforms.Compose([
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
            ])
        else:
            self.transform_center = transforms.Compose([
                transforms.Resize(self.trainsize),
                transforms.ToTensor(),
            ])

    def __getitem__(self, index):

        img = self.data_list['waferMap'][index]

        img = Image.fromarray(img)
        img_torch = self.transform_center(img)

        label = self.data_list['failureNum'][index]

        return img_torch, label

    def __len__(self):
        return self.size


class WM811KDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.classes = sorted(os.listdir(data_path))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images, self.labels = self.load_data()

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        self.transform = transform

    def load_data(self):
        images, labels = [], []
        for class_name in self.classes:
            class_path = os.path.join(self.data_path, class_name)
            class_idx = self.class_to_idx[class_name]
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                images.append(image_path)
                labels.append(class_idx)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('L')
        if self.transform:
            image = self.transform(image)

        return image, label
