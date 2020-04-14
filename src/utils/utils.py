"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""

import tensorflow as tf
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
import torch
import torch.nn as nn

def to_categorical(mask, num_classes, channel='channel_first'):
    assert mask.ndim == 4, "mask should have 4 dims"
    if channel != 'channel_first' and channel != 'channel_last':
        assert False, r"channel should be either 'channel_first' or 'channel_last'"
    assert num_classes > 1, "num_classes should be greater than 1"
    unique = np.unique(mask)
    assert len(unique) <= num_classes, "number of unique values should be smaller or equal to the num_classes"
    assert np.max(unique) < num_classes, "maximum value in the mask should be smaller than the num_classes"
    # shape = mask.shape
    #     # h = shape[1]
    #     # w = shape[2]
    #     # if channel == 'channel_first':
    #     #     output = np.zeros((shape[0], num_classes, h, w))
    #     #     for i in range(num_classes):
    #     #         output[:,i] = mask == i
    #     # else:
    #     #     output = np.zeros((shape[0], h, w, num_classes))
    #     #     for i in range(num_classes):
    #     #         output[...,i] = mask == i
    if mask.shape[1] == 1:
        mask = np.squeeze(mask, axis=1)
    if mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    eye = np.eye(num_classes, dtype='uint8')
    output = eye[mask]
    if channel == 'channel_first':
        output = np.moveaxis(output, -1, 1)
    return output

def soft_to_hard_pred(pred, channel_axis=1):
    max_value = np.max(pred, axis=channel_axis, keepdims=True)
    return np.where(pred==max_value, 1, 0)

def remove_files(directory='../weights/*'):
    import os, glob
    files = glob.glob(directory)
    for f in files:
        print(f)
        os.remove(f)
    print("Files removed")


def read_tf(tfrecord_path):
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_path)

    # Create a dictionary describing the features.
    image_feature_description = {
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        'label_vol': tf.io.FixedLenFeature([], tf.string),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    for parser in parsed_image_dataset:
        data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
        data_vol = tf.reshape(data_vol, [256, 256, 3])
        data_vol = tf.slice(data_vol, [0, 0, 0], [256, 256, 3])

        label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
        label_vol = tf.reshape(label_vol, [256, 256, 3])
        label_vol = tf.slice(label_vol, [0, 0, 1], [256, 256, 1])

        image_raw1 = data_vol.numpy()
        image_raw2 = label_vol.numpy()
        return image_raw1, image_raw2


def plot_slices(data_vol, label_vol):
    """
    :return:
    """
    f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))
    for i in range(20):
        intt = np.random.choice(data_vol.shape[0])

        plots[i // 5, i % 5].axis('off')
        plots[i // 5, i % 5].imshow(data_vol[intt, 0, :, :], cmap=plt.cm.bone)
        plots[i // 5, i % 5].imshow(label_vol[intt, 0, :,:], alpha=0.5)
    plt.show()


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in [1,2,3]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img


def resize_volume(img_volume, w=288, h=288):
    """
    :param img_volume:
    :return:
    """
    img_res = []
    for im in img_volume:
        img_res.append(cv2.resize(im, dsize=(w, h), interpolation=cv2.INTER_NEAREST))

    return np.array(img_res)


def preprocess_volume(img_volume):

    """
    :param img_volume: A patient volume
    :return: applying CLAHE and Bilateral filter for contrast enhacnmeent and denoising

    """
    prepross_imgs = []
    for i in range(len(img_volume)):
        img = img_volume[i]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        cl1 = clahe.apply(img)
        prepross_imgs.append(cl1)

    return np.array(prepross_imgs)



def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, learning_rate):
    lr = lr_poly(learning_rate, i_iter, 250000, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate(optimizer, i_iter, cfg, learning_rate=2.5e-4):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, learning_rate=learning_rate)


def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, learning_rate=1e-4)


if __name__ == '__main__':

    pred = np.random.rand(2, 3, 3)
    print(pred)
    print(soft_to_hard_pred(pred, 0))
    input()

    eye = np.eye(3, dtype='uint8')
    mask = np.array([[1,1,1,1],[1,2,2,1],[1,2,3,1],[1,1,1,1]]) - 1
    print(mask)
    mask1 = np.array([[2,2,2,2],[1,1,2,2],[1,1,1,1],[3,3,3,3]]) - 1
    print(mask1)
    mask = np.array([mask, mask1])
    mask = to_categorical(mask=mask, num_classes=3, channel='channel_first')
    input()
