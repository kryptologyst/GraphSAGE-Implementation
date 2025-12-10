"""Main training script for GraphSAGE implementation."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models import GraphSAGE
from src.data import load_dataset, generate_synthetic_graph, get_graph_statistics
from src.train import NodeClassificationLoss, MetricsTracker, train_epoch, evaluate
from src.eval import evaluate_model_performance, visualize_embeddings, plot_confusion_matrix
from src.utils import set_seed, get_device, count_parameters


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration."""
    if config.logging.use_wandb:
        import wandb
        wandb.init(
            project=config.logging.wandb_project,
            config=OmegaConf.to_container(config, resolve=True),
        )
    
    if config.logging.use_tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        log_dir = Path(config.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir)
    
    return None


def load_data(config: Dict[str, Any]) -> tuple:
    """Load and prepare data."""
    if config.data.dataset == "synthetic":
        data = generate_synthetic_graph(
            num_nodes=config.data.synthetic.num_nodes,
            num_classes=config.data.synthetic.num_classes,
            num_features=config.data.synthetic.num_features,
            edge_prob=config.data.synthetic.edge_prob,
            seed=config.seed,
        )
        dataset_name = "synthetic"
    else:
        data, dataset_name = load_dataset(
            name=config.data.dataset,
            root=config.data.root,
        )
    
    # Print dataset statistics
    stats = get_graph_statistics(data)
    print(f"\nDataset: {dataset_name}")
    print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
    print(f"Features: {stats['num_features']}, Classes: {stats['num_classes']}")
    print(f"Train/Val/Test: {stats['train_size']}/{stats['val_size']}/{stats['test_size']}")
    print(f"Class distribution: {[f'{p:.3f}' for p in stats['class_distribution']]}")
    
    return data, dataset_name


def create_model(config: Dict[str, Any], data: torch.Tensor) -> GraphSAGE:
    """Create GraphSAGE model."""
    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=config.model.hidden_channels,
        out_channels=data.num_classes,
        num_layers=config.model.num_layers,
        aggregator=config.model.aggregator,
        dropout=config.model.dropout,
        use_batch_norm=config.model.use_batch_norm,
        use_residual=config.model.use_residual,
        activation=config.model.activation,
    )
    
    print(f"\nModel created with {count_parameters(model):,} parameters")
    return model


def create_optimizer_and_scheduler(
    model: GraphSAGE,
    config: Dict[str, Any],
) -> tuple:
    """Create optimizer and learning rate scheduler."""
    if config.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
    
    # Learning rate scheduler
    scheduler = None
    if config.training.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.scheduler_params.step_size,
            gamma=config.training.scheduler_params.gamma,
        )
    elif config.training.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
        )
    elif config.training.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=10,
        )
    
    return optimizer, scheduler


def train_model(
    model: GraphSAGE,
    data: torch.Tensor,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Train the GraphSAGE model."""
    # Move data to device
    data = data.to(device)
    
    # Create optimizer and scheduler
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    
    # Create loss function
    criterion = NodeClassificationLoss(
        loss_type=config.loss.type,
        class_weights=config.loss.class_weights,
        label_smoothing=config.loss.label_smoothing,
    )
    
    # Create metrics tracker
    metrics_tracker = MetricsTracker(data.num_classes)
    
    # Setup logging
    writer = setup_logging(config)
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print(f"\nStarting training for {config.training.epochs} epochs...")
    
    for epoch in tqdm(range(1, config.training.epochs + 1), desc="Training"):
        # Train epoch
        train_loss, train_metrics = train_epoch(
            model, data, optimizer, criterion, device, metrics_tracker
        )
        
        # Evaluate
        val_loss, val_metrics = evaluate(
            model, data, criterion, device, "val", metrics_tracker
        )
        
        # Update learning rate
        if scheduler is not None:
            if config.training.scheduler == "plateau":
                scheduler.step(val_metrics.get('accuracy', 0))
            else:
                scheduler.step()
        
        # Log metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_metrics.get('accuracy', 0))
        
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Val', val_metrics.get('accuracy', 0), epoch)
            writer.add_scalar('F1_Macro/Val', val_metrics.get('f1_macro', 0), epoch)
        
        if config.logging.use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_metrics.get('accuracy', 0),
                'val_f1_macro': val_metrics.get('f1_macro', 0),
            })
        
        # Print progress
        if epoch % config.logging.log_interval == 0:
            print(
                f"Epoch {epoch:03d}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_metrics.get('accuracy', 0):.4f}, "
                f"Val F1: {val_metrics.get('f1_macro', 0):.4f}"
            )
        
        # Early stopping
        if config.training.early_stopping.patience > 0:
            if val_metrics.get('accuracy', 0) > best_val_acc + config.training.early_stopping.min_delta:
                best_val_acc = val_metrics.get('accuracy', 0)
                patience_counter = 0
                
                # Save best model
                if config.logging.save_checkpoints:
                    checkpoint_dir = Path(config.logging.checkpoint_dir)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_accuracy': best_val_acc,
                        'config': config,
                    }, checkpoint_dir / 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= config.training.early_stopping.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        # Reset metrics for next epoch
        metrics_tracker.reset()
    
    # Load best model
    if config.logging.save_checkpoints:
        checkpoint_path = Path(config.logging.checkpoint_dir) / 'best_model.pt'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    if writer is not None:
        writer.close()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train GraphSAGE model')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Override dataset name')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override config with command line arguments
    if args.dataset is not None:
        config.data.dataset = args.dataset
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.lr is not None:
        config.training.lr = args.lr
    if args.seed is not None:
        config.seed = args.seed
    
    # Set random seed
    set_seed(config.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    data, dataset_name = load_data(config)
    
    # Create model
    model = create_model(config, data)
    model = model.to(device)
    
    # Train model
    training_results = train_model(model, data, config, device)
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    
    final_results = evaluate_model_performance(model, data, device)
    
    for split, metrics in final_results.items():
        print(f"\n{split.upper()} Results:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Additional analysis
    if config.evaluation.visualize_embeddings:
        print("\nGenerating embedding visualization...")
        visualize_embeddings(model, data, device, save_path="assets/embeddings.png")
    
    if config.evaluation.plot_confusion_matrix:
        print("\nGenerating confusion matrix...")
        plot_confusion_matrix(model, data, device, save_path="assets/confusion_matrix.png")
    
    print(f"\nTraining completed! Best validation accuracy: {training_results['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
