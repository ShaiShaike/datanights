import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.random import choice
from typing import List
from .math_utils import weights2np_probs, choose_index_by_prob


def get_classes_indexes(dataset: Dataset, num_classes: int) -> List[list]:
    classes_indexes = [[] for i in range(num_classes)]
    for i, (_, target) in enumerate(dataset):
        classes_indexes[target].append(i)
    return classes_indexes
    


class WeightedDataset(Dataset):
    def __init__(self, dataset: Dataset, classes_weights: list):
        self.dataset = dataset
        self.classes_indexes = get_classes_indexes(dataset, num_classes=len(classes_weights))
        self.classes_lens = [len(indexes) for indexes in self.classes_indexes]
        self.probabilities = weights2np_probs(classes_weights)

    def __getitem__(self, index):
        chosen_class = choose_index_by_prob(self.probabilities)
        return self.dataset[self.classes_indexes[chosen_class][index % self.classes_lens[chosen_class]]]


    def __len__(self):
        return max(self.classes_lens)
    
    def change_weights(self, classes_weights: list):
        self.classes_indexes = get_classes_indexes(self.dataset, num_classes=len(classes_weights))
        self.classes_lens = [len(indexes) for indexes in self.classes_indexes]
        self.probabilities = weights2np_probs(classes_weights)


def evaluate(net, testloader, classes):
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1


    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                    accuracy))