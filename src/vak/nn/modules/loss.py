import torch
from torch.nn import Module


from .. import functional as F


__all__ = [
    'CombinedCESERLoss',
    'SegmentErrorRateLoss'
]


class SegmentErrorRateLoss(Module):
    def __init__(self, unlabeled=0):
        super(SegmentErrorRateLoss, self).__init__()

        self.unlabeled = unlabeled

    def forward(self, input, target):
        dists = torch.zeros(target.shape[0]).to(target.device)

        input = input.argmax(dim=1)
        input = torch.unbind(input)
        target = torch.unbind(target)
        input = [
            F.lbl_tb2labels(inp, unlabeled=self.unlabeled)
            for inp in input
        ]
        target = [
            F.lbl_tb2labels(targ, self.unlabeled)
            for targ in target
        ]

        for ind, (src, tgt) in enumerate(zip(input, target)):
            dists[ind] = F.segment_error_rate(src, tgt)

        return dists.mean()


class CombinedCESERLoss(Module):
    def __init__(self,
                 lambda_ce=0.5,
                 lambda_ser=0.5,
                 unlabeled=0):
        super(CombinedCESERLoss, self).__init__()

        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.ser_criterion = SegmentErrorRateLoss(unlabeled=unlabeled)

        self.lambda_ce = lambda_ce
        self.lambda_ser = lambda_ser

    def forward(self, input, target):
        ce_loss = self.ce_criterion(input, target)
        ser_loss = self.ser_criterion(input, target)
        return (self.lambda_ce * ce_loss) + (self.lambda_ser * ser_loss)
