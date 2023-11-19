import numpy as np
import torch


class LRAdjustCallback:
    def __init__(self, optimizer, scheduler, patience=7, epsilon=1e-3):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.epsilon = epsilon
        self.counter = 0
        self.best_loss = np.inf

    def step(self, loss):
        if loss + self.epsilon < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n[Callback] Adjusting learning rate. Counter: {self.counter}")
                self.adjust_learning_rate()
                self.counter = 0

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= 0.5
        self.scheduler.last_epoch = -1


class CkptCallback:
    def __init__(self, model, path="model_state.pt"):
        self.best_loss = np.inf
        self.path = path
        self.model = model

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(self.model.state_dict(), self.path)
