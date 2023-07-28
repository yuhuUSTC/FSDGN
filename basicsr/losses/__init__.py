from copy import deepcopy

from basicsr.utils import get_root_logger
from basicsr.utils.registry import LOSS_REGISTRY
from .losses import (AmplitudeLoss, PhaseLoss, CharbonnierLoss, GANLoss, L1Loss, MSELoss, PerceptualLoss, WeightedTVLoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty, ContrastLoss, PreLoss, ColorLoss, LabLoss, GradientLoss, EdgeLoss, SSIMLoss)

__all__ = [
    'AmplitudeLoss', 'PhaseLoss', 'L1Loss', 'MSELoss', 'CharbonnierLoss', 'WeightedTVLoss', 'GANLoss', 'gradient_penalty_loss',
    'r1_penalty', 'g_path_regularize', 'ContrastLoss', 'PreLoss', 'ColorLoss', 'LabLoss', 'GradientLoss', 'EdgeLoss', 'SSIMLoss'
]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
