import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from torch.utils.data.sampler import BatchSampler


class MyDataset(Dataset):
    def __init__(self, data_dir, train=True, selected_idx=None):
        self.train=train
        base_dataset = torchvision.datasets.ImageFolder(data_dir)
        imgs = np.array(base_dataset.imgs)
        img_path, img_class = imgs[:, 0], imgs[:, 1]
        img_class = np.array([int(element) for element in img_class])
        # if train:
        #     self.imgs, self.targets = self.split_data(img_path, img_class, selected_idx)
        # else:
        #     self.imgs, self.targets = img_path, img_class
        self.imgs, self.targets = img_path, img_class
        self.transform = self.get_transform(224)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path).convert('RGB')
        target = self.targets[idx]
        
        if self.transform is not None:
            img=self.transform(img)
        return img, target
       
    def __len__(self):
        return len(self.imgs)

    def get_transform(self, crop_size):
        if self.train:
            return transforms.Compose([transforms.Resize(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(crop_size, padding=int(crop_size*0.125), padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        else:
            return transforms.Compose([transforms.Resize(crop_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    def split_data(self, data, target, sampled_idx):
        return data[sampled_idx], target[sampled_idx]


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        self.labels = dataset.targets
        self.classes =list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels== label)[0]
                                 for label in self.classes}
        for l in self.classes:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.classes}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.classes, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size

