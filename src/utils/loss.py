"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from math import log

torch.set_default_tensor_type('torch.cuda.FloatTensor')
class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        return F.cross_entropy(
            predict, target, weight=weight, size_average=self.size_average
        )


class DiceCoefMultilabelLoss(nn.Module):

    def __init__(self, cuda=True):
        super().__init__()
        # self.smooth = torch.tensor(1., dtype=torch.float32)
        self.one = torch.tensor(1., dtype=torch.float32).cuda()
        self.activation = torch.nn.Softmax2d()

    def dice_loss(self, predict, target):
        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)
        intersection = predict * target.cuda().float()
        score = (intersection.sum() * 2. + 1.) / (predict.sum() + target.sum() + 1.)
        return 1. - score

    def forward(self, predict, target, numLabels=3, channel='channel_first'):
        assert channel in [
            'channel_first',
            'channel_last',
        ], r"channel has to be either 'channel_first' or 'channel_last'"
        dice = 0
        predict = self.activation(predict)
        if channel == 'channel_first':
            for index in range(numLabels):
                #Lme = [0.1, 0.4, 0.2]
                temp = self.dice_loss(predict[:, index, :, :], target[:, index, :, :])
                dice += temp
        else:
            for index in range(numLabels):
                #Lme = [0.1, 0.4, 0.2]
                temp = self.dice_loss(predict[:, :, :, index], target[:, :, :, index])
                dice += temp
        dice = dice / numLabels
        return dice
