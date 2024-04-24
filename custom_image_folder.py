import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random
import os
import torchvision.transforms.functional as TF
from PIL import Image


class CustomImageFolder(datasets.ImageFolder):

    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.root = root
        self.imgs = self.samples
        # Build an auxiliary structure to fetch items by label quickly
        self.label_to_indices = self._map_labels_to_indices()
        self.local_crops = 8
        global_crops_scale=(0.4, 1.)
        local_crops_scale=(0.05, 0.4)

        # Right now, all of these do a random crop, random horizontal flip, then normalize
        self.global_crop_1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.global_crop_2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.local_crop = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _map_labels_to_indices(self):
        label_to_indices = {}
        for idx, (_, label) in enumerate(self.imgs):
            if label in label_to_indices:
                label_to_indices[label].append(idx)
            else:
                label_to_indices[label] = [idx]
        return label_to_indices

    def get_random_subset_for_label(self, label, subset_size, exclude_index=None):
        indices = self.label_to_indices[label].copy()  # Make a copy so we don't modify the original list
        if exclude_index is not None:
            try:
                indices.remove(exclude_index)
            except ValueError:
                pass  # The exclude_index is not in the list, do nothing
        subset_indices = random.sample(indices, min(subset_size, len(indices)))  # Make sure not to exceed the number of samples available
        #return [(self.loader(self.imgs[i][0]), self.imgs[i][1]) for i in subset_indices]
        return subset_indices

    def __getitem__(self, index):
        crops = []
        path, target = self.imgs[index]
        crops.append(self.global_crop_1(self.loader(path))) #global crop original image

        subset_indices = self.get_random_subset_for_label(target, self.local_crops+1, exclude_index=index)

        crops.append(self.global_crop_2(self.loader(self.imgs[subset_indices[0]][0]))) #global crop first subset image

        #local crops
        for i in range(1, len(subset_indices)):
            crops.append(self.local_crop(self.loader(self.imgs[subset_indices[i]][0])))

        return crops
    
    def view_crops(self, crops):
        for j, image in enumerate(crops):
            print(image.size())
            filename = os.path.join('crops', f'crop_{j}.png')
            image = TF.to_pil_image(image)
            image.save(filename)


class CustomImageFolder2(datasets.ImageFolder):
    
    def __init__(self, root, local_crops, crops):
        super().__init__(root, transform=None)
        self.root = root
        self.crops = crops
        self.imgs = self.samples        # list of samples with form (path to sample, class)
        # Build an auxiliary structure to fetch items by label quickly
        self.label_to_indices = self._map_labels_to_indices()
        self.local_crops = local_crops

    def _map_labels_to_indices(self):
        label_to_indices = {}
        for idx, (_, label) in enumerate(self.imgs):
            if label in label_to_indices:
                label_to_indices[label].append(idx)
            else:
                label_to_indices[label] = [idx]
        return label_to_indices

    def get_random_subset_for_label(self, label, subset_size, exclude_index=None):
        indices = self.label_to_indices[label].copy()  # Make a copy so we don't modify the original list
        if exclude_index is not None:
            try:
                indices.remove(exclude_index)
            except ValueError:
                pass  # The exclude_index is not in the list, do nothing
        subset_indices = random.sample(indices, min(subset_size, len(indices)))  # Make sure not to exceed the number of samples available
        #return [(self.loader(self.imgs[i][0]), self.imgs[i][1]) for i in subset_indices]
        return subset_indices

    def __getitem__(self, index):
        crops = list()
        path, target = self.imgs[index]
        # global crops
        crops.append(self.crops["global"](self.loader(path)))
        subset_indices = self.get_random_subset_for_label(target, self.local_crops + 1, exclude_index=index)
        crops.append(self.crops["global"](self.loader(self.imgs[subset_indices[0]][0]))) # global crop first subset image

        #local crops
        for i in range(1, len(subset_indices)):
            crops.append(self.crops["local"](self.loader(self.imgs[subset_indices[i]][0])))
        return crops
    
    def view_crops(self, crops, reverse_transform):
        crops = [reverse_transform(crop) for crop in crops]
        cols = int(len(crops) / 2)
        fig, axs = plt.subplots(2, cols)
        axs = axs.flatten()
        for i, ax in enumerate(axs):
            ax.imshow(crops[i][0])
        plt.show()

if __name__ == "__main__":
    # Assuming '/path/to/dataset' is your directory with class subdirectories
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    global_crop = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.4, 1.), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    local_crop = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.05, 0.4), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    reverse_transform = transforms.Compose([
            transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
            transforms.Lambda(lambda t: t.permute(0, 2, 3, 1)), # CHW to HWC
            transforms.Lambda(lambda t: t.detach().cpu().numpy())
    ])

    crops = {"global":global_crop, "local":local_crop}

    #dataset = CustomImageFolder('data/tiny-imagenet-200/train', transform=None)
    dataset = CustomImageFolder2('../Datasets/imagenet-mini/train', local_crops=8, crops=crops)

    # DataLoader instantiation, nothing different here
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(len(dataloader))

    #expect 10 because the first item in dataset should be the 10 crops
    print(len(dataset[0]))
    # you'll want to comment out the transforms.normalize if you want the images to be interperetable by humans.
    #dataset.view_crops(dataset[0])

    """
    # Example: iterating through DataLoader to get subsets for the first batch
    for imgs, lbls in dataloader:
        # For each image, get a random subset of 5 images with the same label
        subsets = [dataset.get_random_subset_for_label(lbl.item(), subset_size=5, exclude_index=i) for i, lbl in enumerate(lbls)]
        # do something with these subsets, for example, print the size of the first subset
        class_to_idx = dataset.class_to_idx
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        folder_name = [idx_to_class[lbls[j].item()] for j in range(len(lbls))]
        print(folder_name)
        print(len(subsets[0]))
        breakpoint()
    """
    for crops in dataloader:
        dataset.view_crops(crops, reverse_transform=reverse_transform) # only works with batch size = 1

