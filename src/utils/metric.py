"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""
import numpy as np
from medpy.metric.binary import hd, dc, asd

def dice_coef(y_true, y_pred):
    """
    :param y_true:
    :param y_pred:
    :return:
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + 1.0) / (np.sum(y_true) + np.sum(y_pred) + 1.0)


def dice_coef_multilabel(y_true, y_pred, numLabels=3, channel='channel_first'):
    """
    :param y_true:
    :param y_pred:
    :param numLabels:
    :return:
    """
    assert channel in [
        'channel_first',
        'channel_last',
    ], r"channel has to be either 'channel_first' or 'channel_last'"
    dice = 0
    if channel == 'channel_first':
        y_true = np.moveaxis(y_true, 1, -1)
        y_pred = np.moveaxis(y_pred, 1, -1)
    for index in range(numLabels):
        temp = dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
        dice += temp

    dice = dice / (numLabels)
    return dice


def hausdorff_multilabel(y_true, y_pred, numLabels=4, channel='channel_first'):
    """
    :param y_true:
    :param y_pred:
    :param numLabels:
    :return:
    """
    assert channel in [
        'channel_first',
        'channel_last',
    ], r"channel has to be either 'channel_first' or 'channel_last'"
    hd_score = 0
    if channel == 'channel_first':
        y_true = np.moveaxis(y_true, 1, -1)
        y_pred = np.moveaxis(y_pred, 1, -1)
    for index in range(1, numLabels):
        temp = hd(reference=y_true[:, :, :, index], result=y_pred[:, :, :, index])
        hd_score += temp

    hd_score = hd_score / (numLabels - 1)
    return hd_score