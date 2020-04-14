"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""
import numpy as np
from torch import save


class EarlyStoppingCallback:

    def __init__(self, patience, mode="min"):
        assert mode=="max" or mode=="min", "mode can only be /'min/' or /'max/'"
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_result = np.Inf if mode=='min' else np.NINF

    def step(self, monitor):
        # check whether the current loss is lower than the previous best value.
        better = False
        if self.mode=="max":
            better = monitor > self.best_result
        else:
            better = monitor < self.best_result
        # if not count up for how long there was no progress
        if better:
            self.counter = 0
        else:
            self.counter += 1

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        return self.counter >= self.patience


class ModelCheckPointCallback:

    def __init__(self, mode="min", model_name="../weights/model_checkpoint.pt", entire_model=False):
        assert mode=="max" or mode=="min", "mode can only be /'min/' or /'max/'"
        self.mode = mode
        self.best_result = np.Inf if mode=='min' else np.NINF
        self.model_name = model_name
        self.entire_model = entire_model
        self.epoch = 0

    def step(self, monitor, model, epoch):
        # check whether the current loss is lower than the previous best value.
        better = False
        if self.mode=="max":
            better = monitor > self.best_result
        else:
            better = monitor < self.best_result
        # if not count up for how long there was no progress
        if better:
            self.best_result = monitor
            self.epoch = epoch
            if self.entire_model:
                to_save = model
            else:
                to_save = model.state_dict()
            save(to_save, self.model_name)
