import torch

def to_class_indices(tensor, num_classes):
    """
    Convert a tensor to class indices if it has an extra channel dimension.
    
    If the tensor has shape [B, C, H, W] where C equals num_classes, it is assumed to be
    either logits or one-hot encoded. In that case, apply argmax over the channel dimension.
    Otherwise, return the tensor as is.
    """
    if tensor.ndim == 4 and tensor.shape[1] == num_classes:
        return torch.argmax(tensor, dim=1)
    return tensor

def compute_iou(preds, targets, num_classes, eps=1e-6):
    """
    Compute the Intersection over Union (IoU) for each class and return both the mean IoU
    and a dictionary with per-class IoU values.
    
    Args:
        preds (torch.Tensor): Predicted segmentation map (either indices [B, H, W] or 
                              [B, C, H, W] logits/one-hot).
        targets (torch.Tensor): Ground truth segmentation map (either indices [B, H, W] or 
                                [B, C, H, W] one-hot).
        num_classes (int): Number of classes.
        eps (float): Small value to avoid division by zero.
    
    Returns:
        tuple: (mean_iou, iou_per_class) where:
            - mean_iou (torch.Tensor): Mean IoU over all classes.
            - iou_per_class (dict): Mapping from class index to IoU.
    """
    # Convert to class indices if needed.
    preds = to_class_indices(preds, num_classes)
    targets = to_class_indices(targets, num_classes)
    
    # Ensure the shapes match
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds shape {preds.shape} vs targets shape {targets.shape}")
    
    iou_per_class = {}
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (targets == cls)
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum()
        # Avoid division by zero: if union is zero, set IoU to 1.0 for that class
        if union > 0:
            iou = (intersection + eps) / (union + eps)
        else:
            iou = torch.tensor(1.0, device=preds.device)
        iou_per_class[cls] = iou
    mean_iou = torch.stack(list(iou_per_class.values())).mean()
    return mean_iou, iou_per_class

def compute_dice(preds, targets, num_classes, eps=1e-6):
    """
    Compute the Dice coefficient for each class and return both the mean Dice score
    and a dictionary with per-class Dice values.
    
    Args:
        preds (torch.Tensor): Predicted segmentation map (either indices [B, H, W] or 
                              [B, C, H, W] logits/one-hot).
        targets (torch.Tensor): Ground truth segmentation map (either indices [B, H, W] or 
                                [B, C, H, W] one-hot).
        num_classes (int): Number of classes.
        eps (float): Small value to avoid division by zero.
    
    Returns:
        tuple: (mean_dice, dice_per_class) where:
            - mean_dice (torch.Tensor): Mean Dice score over all classes.
            - dice_per_class (dict): Mapping from class index to Dice score.
    """
    # Convert to class indices if needed.
    preds = to_class_indices(preds, num_classes)
    targets = to_class_indices(targets, num_classes)
    
    # Ensure the shapes match
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds shape {preds.shape} vs targets shape {targets.shape}")
    
    dice_per_class = {}
    for cls in range(num_classes):
        pred_inds = (preds == cls).float()
        target_inds = (targets == cls).float()
        intersection = (pred_inds * target_inds).sum()
        dice = (2 * intersection + eps) / (pred_inds.sum() + target_inds.sum() + eps)
        dice_per_class[cls] = dice
    mean_dice = torch.stack(list(dice_per_class.values())).mean()
    return mean_dice, dice_per_class