import torch
from torch.utils.data.sampler import Sampler


class RandomOverSampler(Sampler):
    '''Randomly samples elements up to num_samples. If num_samples is greater
    than the length of the data_source, it oversamples to cover the difference.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default:len(data_source)
    '''

    def __init__(self, data_source, num_samples=None):
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             f"value, but got num_samples={num_samples}")
        self.data_source = data_source
        self.num_samples = len(data_source) if num_samples is None else num_samples

    def __len__(self):
        return self.num_samples

    def __iter__(self):

        # if we need more samples than in data_source
        if self.num_samples > len(self.data_source):
            # get all original samples
            base_samples = torch.randperm(len(self.data_source)).tolist()
            # oversample additional items to reach num_samples
            difference = self.num_samples - len(self.data_source)
            extra_samples = torch.randperm(difference).tolist()
            return iter(base_samples+extra_samples)
        else:
            return iter(torch.randperm(self.num_samples).tolist())
