import numpy as np 
import torch 
import json

"""
This code is from https://github.com/Bjarten/early-stopping-pytorch
"""
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path="./weights/checkpoint.pt", trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0 
        self.best_score = None 
        self.early_stop = False 
        self.early_stop_epoch = 0
        self.val_loss_min = np.Inf 
        self.delta = delta 
        self.path = path 
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model, epoch) ->None :
        score = -val_loss 
        if self.best_score is None:
            self.best_score = score 
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter+=1 
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True 
        else:
            self.best_score = score 
            self.save_checkpoint(val_loss, model)
            self.early_stop_epoch = epoch
            self.counter = 0 
    
    def save_checkpoint(self, val_loss, model) -> None:
        """
        INFO:
            Save model when Validation loss decreased
        """
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# Voting Weights depends on validation losses
# validLoss = {
#   FOLDNAME(str): LOSS(float)
# }
class Voting:
    def __init__(self, jsonFilePath:str):
        self.votingWeights = {}

        with open(jsonFilePath, mode='r') as f:
            validLoss = json.load(f)

        # transform valid loss to Integer
        for fold in validLoss.keys():
            validLoss[fold] = int(validLoss[fold])

        validLossLCD = self._Min_common_multiple(*validLoss.values()) # 計算LCD

        for fold in validLoss.keys():
            self.votingWeights[fold] = (1/validLoss[fold])*validLossLCD # 倒數*LCD

        sumOfVotingWeights = sum(self.votingWeights.values()) # 計算新的總和

        for fold in self.votingWeights.keys():
            self.votingWeights[fold] = self.votingWeights[fold]/sumOfVotingWeights # 使總和為1


    def __call__(self, fold:str) -> float:
        return self.votingWeights[fold]


    def _Commom_multiple(self, num1, num2):
        while num1%num2!=0:
            num1, num2 = num2, (num1%num2)
        return num2


    def _Min_common_multiple(self, *nums):
        while len(nums)>1:
            nums = [nums[i]*nums[i+1]/self._Commom_multiple(nums[i], nums[i+1]) for i in range(len(nums)-1)]
        return int(nums[0])