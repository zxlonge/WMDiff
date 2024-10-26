import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GumbleSoftmax(nn.Module):
    def __init__(self, hard=True):
        super(GumbleSoftmax, self).__init__()
        self.training = False
        self.hard = hard
        self.gpu = False
        self.minval = torch.tensor(0, dtype=torch.float32)
        self.maxval = torch.tensor(1, dtype=torch.float32)

    def cuda(self):
        self.gpu = True

    def cpu(self):
        self.gpu = False

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = torch.rand(template_tensor.shape, dtype=torch.float32)
        uniform_samples_tensor = torch.abs(uniform_samples_tensor)
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits):
        logits.to(device)
        gumble_samples_tensor = self.sample_gumbel_like(logits).to(device)
        gumble_trick_log_prob_samples = logits + gumble_samples_tensor
        soft_samples = F.softmax(gumble_trick_log_prob_samples, dim=-1)
        return soft_samples

    def gumbel_softmax(self, logits, hard=True):
        logits.to(device)
        y = self.gumbel_softmax_sample(logits)
        y_hard = []
        max_value_indexes = []
        if hard:
            max_value_indexes = torch.argmax(y, dim=1)
            y_hard = torch.zeros_like(logits)
            for batch_idx in range(logits.shape[0]):
                y_hard[batch_idx][max_value_indexes[batch_idx]] = 1.0
        return y_hard

    def forward(self, logits, temp=1, force_hard=True):
        result = 0
        if self.training and not force_hard:
            result = self.gumbel_softmax(logits, hard=False)
        else:
            result = self.gumbel_softmax(logits, hard=True)
        return result
