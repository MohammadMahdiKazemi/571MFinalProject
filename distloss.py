"""
Implements the knowledge distillation loss
"""
import torch
from torch.nn import functional as F

class DistillationLoss(torch.nn.Module):
    """
    This module wraps a standard criterion and adds an extra knowledge distillation loss by
    taking a teacher model prediction and using it as additional supervision.
    """
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 alpha: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = alpha

    def forward(self, inputs, outputs, labels):
        """
        Args:
            inputs: The original inputs that are feed to the teacher model
            outputs: the outputs of the model to be trained. It is expected to be
                either a Tensor, or a Tuple[Tensor, Tensor], with the original output
                in the first position and the distillation predictions as the second output
            labels: the labels for the base criterion
        """

        base_loss = self.base_criterion(outputs.squeeze(), labels)
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        imitationloss = self.base_criterion(teacher_outputs.squeeze(),outputs.squeeze())
        teacher_loss = self.base_criterion(teacher_outputs.squeeze(),labels)
        
        phi = 1. - teacher_loss/100.
        
        print("imitation loss",imitationloss,"base loss", base_loss,"phi",phi)
        loss = base_loss * (1 - self.alpha) + (imitationloss * phi) * self.alpha
        return torch.mean(loss), torch.mean(base_loss)