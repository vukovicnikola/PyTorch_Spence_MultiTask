import bz2
import pickle
import _pickle as cPickle
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data.dataloader import default_collate


# Not needed with new LetterBatchSampler and RandomOverSampler
def custom_collate(batch):
    '''Override default collate function to ignore bad samples.
    i.e. if a not all samples are from the same class.'''

    current_class = batch[0]['letter_class']
    out_batch = []
    bad_samples = 0
    for i, sample in enumerate(batch):
        if sample['letter_class'] == current_class:
            out_batch.append(sample)
    if len(out_batch) != len(batch):
        bad_samples += 1

    if bad_samples != 0:
        print(f'Incomplete batch - {bad_samples} bad samples skipped...')
    return default_collate(out_batch)


def show_keypoints(sample):
    """Show image with keypoints from sample"""
    image, keypoints = sample['image'], sample['keypoints']

    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)
    if isinstance(keypoints, torch.Tensor):
        keypoints = keypoints.numpy()

    plt.figure()
    plt.imshow(image)
    plt.scatter(keypoints[:, 1], keypoints[:, 0], s=10, marker='.', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()


def show_batch_sample(sample_batched):
    """Show a sample of images with landmarks for a given batch."""
    images_batch = sample_batched['image']
    keypoints_batch = sample_batched['keypoints']

    im_size = images_batch.size(2)
    grid_border_size = 1

    nrow = 5
    plt.figure(figsize=(30, 30))
    grid = utils.make_grid(images_batch[:nrow,:,:], nrow=nrow)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(nrow):
        plt.scatter(keypoints_batch[i, :, 1].numpy() + i * im_size + (i + 1) * grid_border_size, # x
                    keypoints_batch[i, :, 0].numpy() + grid_border_size, # y
                    s=30, marker='.', c='r')

        plt.title('Batch from dataloader')
    plt.axis('off')
    plt.show()


def get_predictions_batch(sample_batched):
    """Show a sample of model predicted landmarks for a given batch."""
    images_batch = sample_batched['image'][:,-3:,:,:].cpu() # only get last 3 channels (RGB)
    keypoints_pred = sample_batched['predicted_keypoints'].cpu()
    keypoints_batch = keypoints_pred.view(-1,len(sample_batched['keypoints'][1]),2)  # reshape
    predicted_classes = sample_batched['predicted_class'].cpu().max(1, keepdim=True)[1]
    real_classes = sample_batched['real_class'].cpu()

    im_size = images_batch.size(2)
    grid_border_size = 2

    nrow = 8
    plt.figure(figsize=(20, 20))
    grid = utils.make_grid(images_batch[:nrow,:,:], nrow=nrow)
    grid = np.clip(grid.numpy(), 0, 1)  # also clip value ranges for matplotlib
    plt.imshow(grid.transpose((1, 2, 0)))
    title = "Real vs Predicted: "

    for i in range(nrow):
        plt.scatter(keypoints_batch[i, :, 1].numpy() + i * im_size + (i + 1) * grid_border_size,  # x
                    keypoints_batch[i, :, 0].numpy() + grid_border_size,  # y
                    s=30, marker='.', c='r')
        predicted_class = predicted_classes[i]
        title = title + f"**Image {i}: {real_classes[i]} vs {predicted_class.item()}**    "

    plt.title(title)
    plt.axis('off')

    return plt


# Saves the "data" with the "title" and adds the .pickle
def full_pickle(title, data):
    pikd = open(title + '.pickle', 'wb')
    pickle.dump(data, pikd)
    pikd.close()


# loads and returns a pickled objects
def loosen(file):
    pikd = open(file, 'rb')
    data = pickle.load(pikd)
    pikd.close()
    return data


# Pickle a file and then compress it into a file with extension
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data
