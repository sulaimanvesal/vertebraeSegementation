"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import torch as t
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import argparse
from utils.utils import soft_to_hard_pred, keep_largest_connected_components
from utils.loss import DiceCoefMultilabelLoss
from model.dilated_unet import Segmentation_model
from utils.callbacks import EarlyStoppingCallback, ModelCheckPointCallback
from utils.metric import dice_coef_multilabel
from dataset import ImageProcessor, DataGenerator


class Trainer:
    def __init__(self, train_path="../input/trainA.csv",
                 test_path="../input/validA.csv",
                 width=256,
                 height=256,  #image size
                 batch_size=16,
                 n_epoch=50,
                 unet_model=None,
                 unet_loss=DiceCoefMultilabelLoss(),
                 unet_lr=0.001,
                 apply_scheduler=True,  # learning rates
                 gaussian_noise=False,
                 transform=False,
                 r_bright=False,
                 r_gamma=False,
                 n_samples=2000,
                 unet_model_name='unet_model_checkpoint.pt',
                 summary_name='../summary/',
                 channel='channel_first'):

        assert channel == 'channel_first' or channel == 'channel_last', r"channel has to be 'channel_first' or ''channel_last"
        self.train_path, self.test_path = train_path, test_path
        self.WIDTH, self.HEIGHT = width, height
        self.BATCH_SIZE = batch_size
        self.noise = gaussian_noise
        self.epochs = n_epoch
        self.n_samples = n_samples
        self.unet_model_name = unet_model_name
        self.to_save_entire_model = False # with model structure
        self.file_to_save_summary = summary_name
        self.channel = channel
        self._apply_transform = transform
        self._r_bright = r_bright
        self._r_gamma = r_gamma
        self.unet_model = unet_model
        self.unet_loss = unet_loss
        self.unet_lr = unet_lr
        self.apply_scheduler = apply_scheduler
        self.unet_optim  = t.optim.Adam(self.unet_model.parameters(), lr=unet_lr, betas=(0.9, 0.99))

    def valid_model(self, data_generator, hd=False):
        self.unet_model.eval()
        dice_list = []
        loss_list = []
        hd_list = []
        with t.no_grad():
            for dataset in data_generator:
                x_batch, y_batch = dataset
                prediction = self.unet_model.forward(t.tensor(x_batch).cuda(), features_out=False)
                l = self.unet_loss.forward(predict=prediction, target=t.tensor(y_batch).cuda(), channel='channel_first')
                loss_list.append(l.item())
                y_pred = soft_to_hard_pred(prediction.cpu().detach().numpy(), 1)
                dice_list.append(dice_coef_multilabel(y_true=y_batch, y_pred=y_pred, channel='channel_first'))
        output = {}
        output["dice"] = np.mean(np.array(dice_list))
        output["loss"] = np.mean(np.array(loss_list))
        if hd:
            output["hd"] = np.mean(np.array(hd_list))
        return output

    def get_generators(self, ids_train, ids_valid):
        trainA_generator = DataGenerator(df=ids_train,
                                         channel="channel_first",
                                         apply_noise=True,
                                         phase="train",
                                         apply_online_aug=False,
                                         batch_size=self.BATCH_SIZE,
                                         n_samples=self.n_samples)
        validA_generator = DataGenerator(df=ids_valid,
                                         channel="channel_first",
                                         apply_noise=False,
                                         phase="valid",
                                         apply_online_aug=False,
                                         batch_size=self.BATCH_SIZE,
                                         n_samples=-1)
        return iter(trainA_generator), iter(validA_generator)

    def tocude(self):
        self.unet_model.cuda()
        self.unet_loss.cuda()

    def togglephase(self, phase="train"):
        assert phase == "train" or phase == "eval"
        if phase == "train":
            self.unet_model.train()
        else:
            self.unet_model.eval()

    def zerograd(self):
        self.unet_optim.zero_grad()

    def togglegrads(self, model="unet", require_grads=True):
        assert model == "unet"
        if model == "unet":
            for param in self.unet_model.parameters():
                param.requires_grad = require_grads

    def step(self):
        self.unet_optim.step()

    def train_epoch(self, trainA_generator):

        unet_loss = []
        unet_dice = []

        self.togglephase(phase="train")
        # train unet
        for dataA in trainA_generator:
            self.zerograd()
            imgA, maskA = dataA
            # print(imgA.shape, maskA.shape)
            # f = plt.figure()
            # f.add_subplot(1, 2, 1)
            # plt.imshow(np.moveaxis(imgA[2], source=0, destination=2), cmap='gray'),
            # plt.title('Spine MR Image')
            # f.add_subplot(1, 2, 2)
            # plt.imshow(np.moveaxis(imgA[2], source=0, destination=2), cmap='gray'),
            # plt.imshow(np.argmax(maskA[2], axis=0), cmap='jet', alpha=0.5)
            # plt.title('Ground Truth Mask')
            # plt.show(block=True)

            # train the unet model
            self.togglegrads(model="unet", require_grads=True)

            l = t.tensor([0], dtype=t.float32).cuda()
            segmentation, btnck = self.unet_model.forward(t.tensor(imgA).cuda(), features_out=True)
            l_segmentation = self.unet_loss.forward(predict=segmentation,
                                             target=t.tensor(maskA).cuda(),
                                             channel=self.channel)
            l += l_segmentation
            if l != t.tensor([0], dtype=t.float32).cuda():
                l.backward()

            self.step()
            y_pred = soft_to_hard_pred(segmentation.cpu().detach().numpy(), 1)
            unet_loss.append(l_segmentation.item())
            unet_dice.append(dice_coef_multilabel(y_true=maskA, y_pred=y_pred, channel=self.channel))

        output = {}
        output["unet_loss"] = np.mean(np.array(unet_loss))
        output["unet_dice"] = np.mean(np.array(unet_dice))
        return output

    def train_model(self, train=True, comments=''):

        # create directory for the weights
        root_directory = '../weights/' + comments + '/'
        if not os.path.exists(root_directory):
            os.mkdir(root_directory)

        # Split the train data into train and validation using image names
        ids_train = ImageProcessor.split_data(self.train_path)
        ids_valid = ImageProcessor.split_data(self.test_path)
        print("Trainining on {} images and validating on {} images...!!".format(len(ids_train), len(ids_valid)))

        trainA_iterator, validA_iterator = self.get_generators(ids_train, ids_valid)
        # convert models and losses to cuda
        self.tocude()

        if self.apply_scheduler:
            unet_scheduler = ReduceLROnPlateau(optimizer=self.unet_optim,
                                               mode='max',
                                               factor=.1,
                                               patience=100,
                                               verbose=True)

        earlystop = EarlyStoppingCallback(patience=10, mode="max")
        modelcheckpoint_unet = ModelCheckPointCallback(mode="max",
                                                       model_name=root_directory + self.unet_model_name,
                                                       entire_model=self.to_save_entire_model)

        train_loss = []
        train_dice = []
        val_loss = []
        val_dice = []

        for epoch in range(self.epochs):
            ###################
            # train the model #
            ###################
            if train:
                print("start to train")
                output = self.train_epoch(trainA_iterator)
                train_loss.append(output["unet_loss"])
                train_dice.append(output["unet_dice"])

                # reduceLROnPlateau
                if self.apply_scheduler:
                    unet_scheduler.step(metrics=train_dice[-1])

            ######################
            # validate and test the model #
            ######################
            self.togglephase(phase="eval")

            print("start to valid")
            output = self.valid_model(data_generator=validA_iterator, hd=False)
            val_dice.append(output["dice"])
            val_loss.append(output["loss"])
            epoch_len = len(str(self.epochs))

            print_msg_line1 = f'valid_loss: {val_loss[-1]:.5f} '
            print_msg_line2 = f'valid_dice: {val_dice[-1]:.5f} '
            if train:
                print_msg_line1 = f'train_loss: {train_loss[-1]:.5f} ' + print_msg_line1
                print_msg_line2 = f'train_dice: {train_dice[-1]:.5f} ' + print_msg_line2
            print_msg_line1 = f'[{epoch + 1:>{epoch_len}}/{self.epochs:>{epoch_len}}] ' + print_msg_line1
            print_msg_line2 = ' ' * (2 * epoch_len + 4) + print_msg_line2
            print(print_msg_line1)
            print(print_msg_line2)

            # model chekc point
            monitor_score = val_dice[-1]
            modelcheckpoint_unet.step(monitor=monitor_score, model=self.unet_model, epoch=epoch)

            # early stop
            earlystop.step(val_dice[-1])
            if earlystop.should_stop():
                break
        the_epoch = modelcheckpoint_unet.epoch
        print("Best model on epoch {}: train_dice {}, valid_dice {}".format(the_epoch,
                                                                            train_dice[the_epoch],
                                                                            val_dice[the_epoch]))
        # record train metrics in tensorboard
        writer = SummaryWriter(comment=comments)
        # writer = SummaryWriter()
        i = 0
        print("write a training summary")
        for t_loss, t_dice, v_loss, v_dice in zip(
                train_loss, train_dice, val_loss, val_dice):
            writer.add_scalar('Loss/Training', t_loss, i)
            writer.add_scalar('Loss/Validation', v_loss, i)
            i += 1
        writer.close()
        print("Finish training")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-unetlr", help="to set the learning rate for the unet", type=float, default=0.0001)
    parser.add_argument("-e", "--epochs", help="the number of epochs to train", type=int, default=300)
    parser.add_argument("-gn", "--gaussianNoise", help="whether to apply gaussian noise", action="store_true",
                        default=True)
    parser.add_argument("--n_samples", help="number of samples to train", type=int, default=100)
    parser.add_argument("-bs", "--batch_size", help="batch size of training", type=int, default=4)
    parser.add_argument("-nc", "--n_class", help="number of classes to segment", type=int, default=3)
    parser.add_argument("-nf", "--n_filter", help="number of initial filters for DR-UNET", type=int, default=32)
    parser.add_argument("-nb", "--n_block", help="number unet blocks", type=int, default=4)
    parser.add_argument("-pt", "--pretrained", help="whether to train from scratch or resume", action="store_true",
                        default=False)
    args = parser.parse_args()

    config_info = "filters {}, n_block {}".format(args.n_filter, args.n_block)
    print(config_info)

    # calculate the comments
    comments = "Verterbra_disk.unet_lr_{}_{}".format(args.unetlr, args.n_filter)
    if args.gaussianNoise:
        comments += ".gaussian_noise"
    print(comments)

    unet_model = Segmentation_model(filters=args.n_filter,
                                    in_channels=3,
                                    n_block=args.n_block,
                                    bottleneck_depth=4,
                                    n_class=args.n_class)

    if args.pretrained:
        unet_model.load_state_dict(torch.load('../weights/{}/unet_model_checkpoint.pt'.format(comments)))

    train_obj = Trainer(width= 256,
                        height=256,
                        batch_size=args.batch_size,  # 8
                        unet_model=unet_model,
                        unet_loss=DiceCoefMultilabelLoss(),
                        gaussian_noise=args.gaussianNoise,
                        unet_lr=args.unetlr,
                        n_epoch=args.epochs,
                        n_samples=args.n_samples)

    # train the models
    print("number of samples {}".format(args.n_samples))
    start = datetime.now()
    t.autograd.set_detect_anomaly(True)
    train_obj.train_model(comments=comments)
    end = datetime.now()
    print("time elapsed for training (hh:mm:ss.ms) {}".format(end - start))