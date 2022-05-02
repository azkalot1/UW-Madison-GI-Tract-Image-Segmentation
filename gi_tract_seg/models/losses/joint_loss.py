from torch.nn.modules.loss import _Loss


class WeightedLoss(_Loss):
    """Wrapper class around loss function that applies weighted with fixed factor.
    This class helps to balance multiple losses if they have different scales
    """

    def __init__(self, loss, weight=1.0):
        super().__init__()
        self.loss = loss
        self.weight = weight

    def forward(self, *input):
        return self.loss(*input) * self.weight


class JointLoss(_Loss):
    def __init__(self, weights, **kwargs):
        super().__init__()
        self.losses = []
        for idx, loss in enumerate(kwargs.values()):
            self.losses.append(WeightedLoss(loss, weights[idx]))

    def forward(self, *input):
        return sum([loss(*input) for loss in self.losses])
