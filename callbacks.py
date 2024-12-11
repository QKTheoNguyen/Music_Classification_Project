import torch
import yaml
import os
from torch.utils.tensorboard import SummaryWriter

class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

class ReduceLROnPlateau:
    def __init__(self, factor=0.1, patience=3):
        self.factor = factor
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_lr = None
        self.reduce_lr = False

    def __call__(self, val_loss, optimizer):
        
        self.reduce_lr = False
        if self.best_score is None:
            self.best_score = val_loss
            self.best_lr = optimizer.param_groups[0]['lr']
        elif val_loss > self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                new_lr = self.best_lr * self.factor
                self.reduce_lr = True
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
                self.counter = 0
        else:
            self.best_score = val_loss
            self.best_lr = optimizer.param_groups[0]['lr']
            self.counter = 0


class ModelCheckpoint:
    def __init__(self, model, save_path):
        self.model = model
        self.save_path = save_path
        self.best_score = None
        self.epoch = None
        self.save = False

    def __call__(self, val_loss, epoch):
        self.save = False
        self.epoch = epoch
        if self.best_score is None or val_loss < self.best_score:
            self.save = True
            self.best_score = val_loss
            torch.save(self.model.state_dict(), os.path.join(self.save_path, "model.pth"))

class TensorBoard:
    def __init__(self, log_dir, config):
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        self.config = config

    def __call__(self, epoch, train_loss, val_loss):
        if epoch == 0:
            self.writer.add_text("Config", yaml.dump(self.config))
            with open(os.path.join(self.log_dir, "config.yaml"), "w") as file:
                yaml.dump(self.config, file)
        self.writer.add_scalar('Loss/Train', train_loss, epoch)
        self.writer.add_scalar('Loss/Validation', val_loss, epoch)

    def close(self):
        self.writer.close()