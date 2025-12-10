"""Training utilities and loss functions for GraphSAGE."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torchmetrics import Accuracy, F1Score, AUROC
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score


class NodeClassificationLoss(nn.Module):
    """Loss function for node classification tasks."""
    
    def __init__(
        self,
        loss_type: str = "cross_entropy",
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        """Initialize loss function.
        
        Args:
            loss_type: Type of loss ('cross_entropy', 'focal', 'label_smoothing').
            class_weights: Class weights for imbalanced datasets.
            label_smoothing: Label smoothing factor.
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        
        if loss_type == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        elif loss_type == "focal":
            self.criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss.
        
        Args:
            logits: Model predictions.
            targets: Ground truth labels.
            
        Returns:
            Loss value.
        """
        return self.criterion(logits, targets)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        """Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class.
            gamma: Focusing parameter.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            inputs: Model predictions.
            targets: Ground truth labels.
            
        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


class MetricsTracker:
    """Track and compute various metrics for node classification."""
    
    def __init__(self, num_classes: int, task: str = "multiclass"):
        """Initialize metrics tracker.
        
        Args:
            num_classes: Number of classes.
            task: Task type ('multiclass', 'multilabel', 'binary').
        """
        self.num_classes = num_classes
        self.task = task
        
        # Initialize metrics
        self.accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.f1_micro = MulticlassF1Score(num_classes=num_classes, average='micro')
        self.f1_macro = MulticlassF1Score(num_classes=num_classes, average='macro')
        self.auroc = AUROC(task=task, num_classes=num_classes)
        
        # Store predictions and targets for epoch-level metrics
        self.predictions = []
        self.targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metrics with new predictions.
        
        Args:
            predictions: Model predictions.
            targets: Ground truth labels.
        """
        self.predictions.append(predictions.detach().cpu())
        self.targets.append(targets.detach().cpu())
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics.
        
        Returns:
            Dictionary of metric values.
        """
        if not self.predictions:
            return {}
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions, dim=0)
        all_targets = torch.cat(self.targets, dim=0)
        
        # Compute metrics
        metrics = {
            'accuracy': self.accuracy(all_preds, all_targets).item(),
            'f1_micro': self.f1_micro(all_preds, all_targets).item(),
            'f1_macro': self.f1_macro(all_preds, all_targets).item(),
            'auroc': self.auroc(all_preds, all_targets).item(),
        }
        
        return metrics
    
    def reset(self) -> None:
        """Reset metrics for new epoch."""
        self.predictions.clear()
        self.targets.clear()
        self.accuracy.reset()
        self.f1_micro.reset()
        self.f1_macro.reset()
        self.auroc.reset()


def train_epoch(
    model: nn.Module,
    data: Data,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    metrics_tracker: Optional[MetricsTracker] = None,
) -> Tuple[float, Dict[str, float]]:
    """Train model for one epoch.
    
    Args:
        model: GraphSAGE model.
        data: Graph data.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to use.
        metrics_tracker: Optional metrics tracker.
        
    Returns:
        Tuple of (average_loss, metrics_dict).
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    # Forward pass
    logits = model(data.x, data.edge_index)
    loss = criterion(logits[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()
    num_batches += 1
    
    # Update metrics
    if metrics_tracker is not None:
        with torch.no_grad():
            preds = F.softmax(logits[data.train_mask], dim=1)
            metrics_tracker.update(preds, data.y[data.train_mask])
    
    avg_loss = total_loss / num_batches
    metrics = metrics_tracker.compute() if metrics_tracker else {}
    
    return avg_loss, metrics


def evaluate(
    model: nn.Module,
    data: Data,
    criterion: nn.Module,
    device: torch.device,
    split: str = "val",
    metrics_tracker: Optional[MetricsTracker] = None,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on validation or test set.
    
    Args:
        model: GraphSAGE model.
        data: Graph data.
        criterion: Loss function.
        device: Device to use.
        split: Split to evaluate ('val' or 'test').
        metrics_tracker: Optional metrics tracker.
        
    Returns:
        Tuple of (loss, metrics_dict).
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        
        if split == "val":
            mask = data.val_mask
        elif split == "test":
            mask = data.test_mask
        else:
            raise ValueError(f"Unknown split: {split}")
        
        loss = criterion(logits[mask], data.y[mask])
        
        # Update metrics
        if metrics_tracker is not None:
            preds = F.softmax(logits[mask], dim=1)
            metrics_tracker.update(preds, data.y[mask])
    
    metrics = metrics_tracker.compute() if metrics_tracker else {}
    
    return loss.item(), metrics


def compute_degree_based_metrics(
    model: nn.Module,
    data: Data,
    device: torch.device,
    num_degree_bins: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics stratified by node degree.
    
    Args:
        model: GraphSAGE model.
        data: Graph data.
        device: Device to use.
        num_degree_bins: Number of degree bins.
        
    Returns:
        Dictionary of metrics per degree bin.
    """
    model.eval()
    
    # Compute node degrees
    degrees = torch.zeros(data.num_nodes, dtype=torch.long)
    for i in range(data.num_nodes):
        degrees[i] = (data.edge_index[0] == i).sum()
    
    # Create degree bins
    degree_bins = torch.linspace(degrees.min(), degrees.max(), num_degree_bins + 1)
    
    results = {}
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits.argmax(dim=1)
        
        for i in range(num_degree_bins):
            bin_start = degree_bins[i]
            bin_end = degree_bins[i + 1]
            
            # Find nodes in this degree bin
            bin_mask = (degrees >= bin_start) & (degrees < bin_end) & data.test_mask
            if bin_mask.sum() == 0:
                continue
            
            bin_predictions = predictions[bin_mask]
            bin_targets = data.y[bin_mask]
            
            # Compute accuracy
            accuracy = (bin_predictions == bin_targets).float().mean().item()
            
            results[f"degree_bin_{i}"] = {
                "degree_range": f"[{bin_start:.1f}, {bin_end:.1f})",
                "num_nodes": bin_mask.sum().item(),
                "accuracy": accuracy,
            }
    
    return results
