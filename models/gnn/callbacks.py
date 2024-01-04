import numpy as np
import torch


class LRAdjustCallback:
    def __init__(self, optimizer, patience=7, epsilon=1e-10, gamma=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.epsilon = epsilon
        self.gamma = gamma
        self.counter = 0
        self.best_loss = np.inf

    def step(self, loss):
        if loss + self.epsilon * loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\n[Callback] Adjusting lr. Counter: {self.counter}\n")
                self.adjust_learning_rate()
                self.counter = 0

    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.gamma


class CkptCallback:
    def __init__(self, model, path="model_state.pt"):
        self.best_loss = np.inf
        self.path = path
        self.model = model

    def step(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            torch.save(self.model.state_dict(), self.path)


class EarlyStoppingCallback:
    def __init__(self, patience=40):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping ....")
