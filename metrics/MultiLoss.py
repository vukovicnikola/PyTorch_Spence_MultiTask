import torch
from torch import nn
from math import log


class MultiLoss(nn.Module):
    '''Calculatess simple summed multi-task loss for batch'''

    def __init__(self, e=2, w=10):
        self.w = w
        self.e = e

    def wing_loss(self, y_true, y_pred):
        y_pred = y_pred.view(-1,len(y_true[1]),2)
        diff = y_true - y_pred
        diff_abs = diff.abs()
        loss = diff_abs.clone()

        case1 = diff_abs < self.w
        case2 = diff_abs >= self.w

        loss[case1] = self.w * torch.log(1 + (loss[case1]/self.e))
        loss[case2] = loss[case2] - (self.w - self.w*log(1.0+self.w/self.e))

        return loss

    def pixel_penalty(self, batch):
        '''Target keypoints should be on the letter stroke, not the background.
        Thus, we increase keypoint loss if the predicted coordinate is on the
        background (nb: white pixels/background are 1 in grayscale images
        and foreground/black ones are 0). '''

        # get maximum image width
        img_dim = batch['image'][0].shape[0]
        keypoints_pred = batch['predicted_keypoints'].view(-1,len(batch['keypoints'][1]),2)  # reshape
        pixel_weights = torch.zeros_like(keypoints_pred)

        for sample_idx, sample in enumerate(keypoints_pred):
            for key_idx, keypoint in enumerate(sample):

                # get predicted x and y coordinates for this keypoint
                # if predicted coord is > img_dim, return img_dim
                x = min(int(keypoint[1].item()), img_dim)
                y = min(int(keypoint[0].item()), img_dim)

                # get pixel value of image at sample_idx, channel 0, XY coords
                pixel_value = batch['image'][sample_idx][0][x, y].item()

                pixel_weights[sample_idx][key_idx][0] = pixel_value
                pixel_weights[sample_idx][key_idx][1] = pixel_value

        return pixel_weights

    def __call__(self, batch):

        c_entropy = nn.CrossEntropyLoss()
        class_loss = c_entropy(batch['predicted_class'], batch['real_class'])
        keypoint_loss = self.wing_loss(batch['keypoints'], batch['predicted_keypoints'])

        #pixel_penalty = self.pixel_penalty(batch)*keypoint_loss

        # Mean reduction for keypoint loss
        mean_keypoint_loss = keypoint_loss.mean()
        #mean_pixel_penalty = pixel_penalty.mean()

        #return class_loss*10 + mean_keypoint_loss + mean_pixel_penalty, class_loss*10, mean_keypoint_loss+mean_pixel_penalty
        return class_loss*10 + mean_keypoint_loss, class_loss*10, mean_keypoint_loss


class LearnedMultiLoss(nn.Module):
    '''Calculatess multi-task loss with a learned sub-task-loss weighting'''
    def __init__(self, task_num=2, e=2, w=10):
        super(LearnedMultiLoss, self).__init__()
        self.w = w
        self.e = e
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def wing_loss(self, y_true, y_pred):
        y_pred = y_pred.view(-1,len(y_true[1]),2)
        diff = y_true - y_pred
        diff_abs = diff.abs()
        loss = diff_abs.clone()

        case1 = diff_abs < self.w
        case2 = diff_abs >= self.w

        loss[case1] = self.w * torch.log(1 + (loss[case1]/self.e))
        loss[case2] = loss[case2] - (self.w - self.w*log(1.0+self.w/self.e))

        loss = loss.mean()
        return loss

    def forward(self, batch):

        c_entropy = nn.CrossEntropyLoss()
        class_loss = c_entropy(batch['predicted_class'], batch['real_class'])
        keypoint_loss = self.wing_loss(batch['keypoints'], batch['predicted_keypoints'])

        precision_class = torch.exp(-self.log_vars[0])
        loss0 = precision_class*class_loss + self.log_vars[0]

        precision_kp = torch.exp(-self.log_vars[1])
        loss1 = precision_kp*keypoint_loss + self.log_vars[1]

        return loss0+loss1, class_loss, keypoint_loss
