import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

import pdb

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.root = root
        self.imgs = self.samples
        # Build an auxiliary structure to fetch items by label quickly
        self.label_to_indices = self._map_labels_to_indices()

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
        return [(self.loader(self.imgs[i][0]), self.imgs[i][1]) for i in subset_indices]


if __name__ == "__main__":
    # Assuming '/path/to/dataset' is your directory with class subdirectories
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = CustomImageFolder('tiny-imagenet-200/tiny-imagenet-200/train', transform=transform)

    # DataLoader instantiation, nothing different here
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(len(dataloader))

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