# AUTOGENERATED! DO NOT EDIT! File to edit: 02_loss.ipynb (unless otherwise specified).

__all__ = ['SDTLoss']

# Cell
from fastai.vision.all import *

# Cell
class SDTLoss(Module):
  def __init__(self, lambda_):
    super().__init__()

    self.lambda_ = lambda_

  def set_path_prob(self, path_prob): self.path_prob = path_prob
  def set_regularizer(self, numers, denoms): self.numers, self.denoms = numers, denoms

  def forward(self, output, target):
    # number of target categories
    target_ohe = torch.zeros((output.shape[0], output.shape[-1])).cuda()

    target = target.view(-1, 1)

    # assert to find out whether shape of target
    # is similar to output or not
    target_ohe.scatter_(1, target, 1).cuda()
    target_ohe = target_ohe.unsqueeze(dim=2)

    log_output = torch.log(output)
    res = torch.bmm(log_output, target_ohe).squeeze(dim=2)

    # weigh cross entropy over all paths
    res = (self.path_prob * res).sum(dim=1)

    # calculate regularizer
    alphas = self.numers.sum(dim=0) / self.denoms.sum(dim=0)
    lambdas_ = torch.ones_like(alphas)

    for i in range(int(np.log2(len(lambdas_) + 1))):
      start_index = 2 ** i - 1
      end_index   = 2 ** i
      lambdas_[start_index:start_index+end_index] = 2 ** (-i)

    lambdas_ = self.lambda_ * lambdas_

    reg_inner_term = (0.5 * torch.log(alphas) + torch.log(1 - alphas) * 0.5) * lambdas_
    C = -reg_inner_term.sum()

    return -res.mean() + C