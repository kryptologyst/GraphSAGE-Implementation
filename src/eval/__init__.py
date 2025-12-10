"""Evaluation utilities for GraphSAGE models."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model_performance(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    splits: List[str] = ["train", "val", "test"],
) -> Dict[str, Dict[str, float]]:
    """Comprehensive model evaluation across different splits.
    
    Args:
        model: Trained GraphSAGE model.
        data: Graph data object.
        device: Device to use.
        splits: List of splits to evaluate.
        
    Returns:
        Dictionary with metrics for each split.
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits.argmax(dim=1)
        probabilities = torch.softmax(logits, dim=1)
        
        for split in splits:
            if split == "train" and hasattr(data, 'train_mask'):
                mask = data.train_mask
            elif split == "val" and hasattr(data, 'val_mask'):
                mask = data.val_mask
            elif split == "test" and hasattr(data, 'test_mask'):
                mask = data.test_mask
            else:
                continue
            
            if mask.sum() == 0:
                continue
            
            split_predictions = predictions[mask]
            split_targets = data.y[mask]
            split_probabilities = probabilities[mask]
            
            # Compute metrics
            accuracy = (split_predictions == split_targets).float().mean().item()
            
            # F1 scores
            f1_micro = compute_f1_score(split_targets, split_predictions, average='micro')
            f1_macro = compute_f1_score(split_targets, split_predictions, average='macro')
            
            # AUROC
            auroc = compute_auroc(split_targets, split_probabilities)
            
            results[split] = {
                'accuracy': accuracy,
                'f1_micro': f1_micro,
                'f1_macro': f1_macro,
                'auroc': auroc,
                'num_samples': mask.sum().item(),
            }
    
    return results


def compute_f1_score(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    average: str = 'macro',
) -> float:
    """Compute F1 score.
    
    Args:
        targets: Ground truth labels.
        predictions: Predicted labels.
        average: Averaging strategy.
        
    Returns:
        F1 score.
    """
    from sklearn.metrics import f1_score
    
    return f1_score(
        targets.cpu().numpy(),
        predictions.cpu().numpy(),
        average=average,
        zero_division=0,
    )


def compute_auroc(
    targets: torch.Tensor,
    probabilities: torch.Tensor,
) -> float:
    """Compute AUROC for multiclass classification.
    
    Args:
        targets: Ground truth labels.
        probabilities: Predicted probabilities.
        
    Returns:
        AUROC score.
    """
    from sklearn.metrics import roc_auc_score
    
    try:
        # For multiclass, use one-vs-rest
        return roc_auc_score(
            targets.cpu().numpy(),
            probabilities.cpu().numpy(),
            multi_class='ovr',
            average='macro',
        )
    except ValueError:
        # Fallback for binary classification
        return roc_auc_score(
            targets.cpu().numpy(),
            probabilities.cpu().numpy(),
        )


def generate_classification_report(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    split: str = "test",
    class_names: Optional[List[str]] = None,
) -> str:
    """Generate detailed classification report.
    
    Args:
        model: Trained model.
        data: Graph data.
        device: Device to use.
        split: Split to evaluate.
        class_names: Optional class names.
        
    Returns:
        Classification report string.
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits.argmax(dim=1)
        
        if split == "test":
            mask = data.test_mask
        elif split == "val":
            mask = data.val_mask
        elif split == "train":
            mask = data.train_mask
        else:
            raise ValueError(f"Unknown split: {split}")
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = predictions[mask].cpu().numpy()
        
        return classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            zero_division=0,
        )


def plot_confusion_matrix(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    split: str = "test",
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot confusion matrix.
    
    Args:
        model: Trained model.
        data: Graph data.
        device: Device to use.
        split: Split to evaluate.
        class_names: Optional class names.
        save_path: Optional path to save plot.
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits.argmax(dim=1)
        
        if split == "test":
            mask = data.test_mask
        elif split == "val":
            mask = data.val_mask
        elif split == "train":
            mask = data.train_mask
        else:
            raise ValueError(f"Unknown split: {split}")
        
        y_true = data.y[mask].cpu().numpy()
        y_pred = predictions[mask].cpu().numpy()
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.title(f'Confusion Matrix - {split.capitalize()} Set')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def visualize_embeddings(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    method: str = "tsne",
    save_path: Optional[str] = None,
) -> None:
    """Visualize learned node embeddings.
    
    Args:
        model: Trained model.
        data: Graph data.
        device: Device to use.
        method: Dimensionality reduction method ('tsne', 'umap').
        save_path: Optional path to save plot.
    """
    model.eval()
    
    with torch.no_grad():
        # Get embeddings (before final classification layer)
        embeddings = model.get_embeddings(data.x, data.edge_index)
        embeddings = embeddings.cpu().numpy()
        labels = data.y.cpu().numpy()
        
        # Dimensionality reduction
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == "umap":
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
            except ImportError:
                print("UMAP not available, falling back to t-SNE")
                reducer = TSNE(n_components=2, random_state=42)
                embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=labels,
            cmap='tab10',
            alpha=0.7,
            s=20,
        )
        plt.colorbar(scatter)
        plt.title(f'Node Embeddings Visualization ({method.upper()})')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def analyze_prediction_confidence(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device,
    split: str = "test",
) -> Dict[str, float]:
    """Analyze prediction confidence and uncertainty.
    
    Args:
        model: Trained model.
        data: Graph data.
        device: Device to use.
        split: Split to analyze.
        
    Returns:
        Dictionary with confidence statistics.
    """
    model.eval()
    
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probabilities = torch.softmax(logits, dim=1)
        
        if split == "test":
            mask = data.test_mask
        elif split == "val":
            mask = data.val_mask
        elif split == "train":
            mask = data.train_mask
        else:
            raise ValueError(f"Unknown split: {split}")
        
        split_probabilities = probabilities[mask]
        
        # Compute confidence metrics
        max_probs = split_probabilities.max(dim=1)[0]
        entropy = -(split_probabilities * torch.log(split_probabilities + 1e-8)).sum(dim=1)
        
        return {
            'mean_confidence': max_probs.mean().item(),
            'std_confidence': max_probs.std().item(),
            'mean_entropy': entropy.mean().item(),
            'std_entropy': entropy.std().item(),
            'min_confidence': max_probs.min().item(),
            'max_confidence': max_probs.max().item(),
        }


def compare_model_performance(
    results: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
) -> None:
    """Compare performance across different models or configurations.
    
    Args:
        results: Dictionary with results from different models.
        metric: Metric to compare.
    """
    model_names = list(results.keys())
    metric_values = [results[name].get(metric, 0) for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values)
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{value:.3f}',
            ha='center',
            va='bottom',
        )
    
    plt.tight_layout()
    plt.show()
