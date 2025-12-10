"""Tests for GraphSAGE implementation."""

import pytest
import torch
import numpy as np

from src.models import GraphSAGE, GraphSAGEPooling
from src.data import generate_synthetic_graph, get_graph_statistics
from src.train import NodeClassificationLoss, MetricsTracker
from src.utils import set_seed, get_device, count_parameters


class TestGraphSAGE:
    """Test GraphSAGE model."""
    
    def test_model_creation(self):
        """Test model creation with different configurations."""
        model = GraphSAGE(
            in_channels=10,
            hidden_channels=[16, 8],
            out_channels=3,
            num_layers=2,
        )
        
        assert model.in_channels == 10
        assert model.out_channels == 3
        assert len(model.convs) == 2
        assert count_parameters(model) > 0
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = GraphSAGE(
            in_channels=10,
            hidden_channels=[16, 8],
            out_channels=3,
        )
        
        # Create synthetic data
        data = generate_synthetic_graph(
            num_nodes=100,
            num_classes=3,
            num_features=10,
        )
        
        # Forward pass
        output = model(data.x, data.edge_index)
        
        assert output.shape == (100, 3)
        assert not torch.isnan(output).any()
    
    def test_different_aggregators(self):
        """Test different aggregation strategies."""
        aggregators = ["mean", "max"]
        
        for agg in aggregators:
            model = GraphSAGE(
                in_channels=10,
                hidden_channels=[16],
                out_channels=3,
                aggregator=agg,
            )
            
            data = generate_synthetic_graph(
                num_nodes=50,
                num_classes=3,
                num_features=10,
            )
            
            output = model(data.x, data.edge_index)
            assert output.shape == (50, 3)
    
    def test_residual_connections(self):
        """Test residual connections."""
        model = GraphSAGE(
            in_channels=10,
            hidden_channels=[16, 8],
            out_channels=3,
            use_residual=True,
        )
        
        data = generate_synthetic_graph(
            num_nodes=50,
            num_classes=3,
            num_features=10,
        )
        
        output = model(data.x, data.edge_index)
        assert output.shape == (50, 3)
    
    def test_batch_normalization(self):
        """Test batch normalization."""
        model = GraphSAGE(
            in_channels=10,
            hidden_channels=[16, 8],
            out_channels=3,
            use_batch_norm=True,
        )
        
        data = generate_synthetic_graph(
            num_nodes=50,
            num_classes=3,
            num_features=10,
        )
        
        output = model(data.x, data.edge_index)
        assert output.shape == (50, 3)


class TestGraphSAGEPooling:
    """Test GraphSAGE with pooling."""
    
    def test_pooling_model(self):
        """Test GraphSAGE with pooling."""
        model = GraphSAGEPooling(
            in_channels=10,
            hidden_channels=[16, 8],
            out_channels=3,
            pooling="mean",
        )
        
        data = generate_synthetic_graph(
            num_nodes=50,
            num_classes=3,
            num_features=10,
        )
        
        # Create batch vector (single graph)
        batch = torch.zeros(50, dtype=torch.long)
        
        output = model(data.x, data.edge_index, batch)
        assert output.shape == (1, 3)
    
    def test_different_pooling_strategies(self):
        """Test different pooling strategies."""
        poolings = ["mean", "max", "add"]
        
        for pooling in poolings:
            model = GraphSAGEPooling(
                in_channels=10,
                hidden_channels=[16],
                out_channels=3,
                pooling=pooling,
            )
            
            data = generate_synthetic_graph(
                num_nodes=50,
                num_classes=3,
                num_features=10,
            )
            
            batch = torch.zeros(50, dtype=torch.long)
            output = model(data.x, data.edge_index, batch)
            assert output.shape == (1, 3)


class TestDataUtilities:
    """Test data utilities."""
    
    def test_synthetic_graph_generation(self):
        """Test synthetic graph generation."""
        data = generate_synthetic_graph(
            num_nodes=100,
            num_classes=5,
            num_features=20,
        )
        
        assert data.num_nodes == 100
        assert data.num_node_features == 20
        assert data.num_classes == 5
        assert hasattr(data, 'train_mask')
        assert hasattr(data, 'val_mask')
        assert hasattr(data, 'test_mask')
    
    def test_graph_statistics(self):
        """Test graph statistics computation."""
        data = generate_synthetic_graph(
            num_nodes=100,
            num_classes=5,
            num_features=20,
        )
        
        stats = get_graph_statistics(data)
        
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'num_features' in stats
        assert 'num_classes' in stats
        assert stats['num_nodes'] == 100
        assert stats['num_features'] == 20
        assert stats['num_classes'] == 5


class TestLossFunctions:
    """Test loss functions."""
    
    def test_cross_entropy_loss(self):
        """Test cross entropy loss."""
        criterion = NodeClassificationLoss(loss_type="cross_entropy")
        
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        
        loss = criterion(logits, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_focal_loss(self):
        """Test focal loss."""
        criterion = NodeClassificationLoss(loss_type="focal")
        
        logits = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        
        loss = criterion(logits, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestMetricsTracker:
    """Test metrics tracker."""
    
    def test_metrics_tracker(self):
        """Test metrics tracker."""
        tracker = MetricsTracker(num_classes=3)
        
        # Add some predictions
        predictions = torch.randn(10, 3)
        targets = torch.randint(0, 3, (10,))
        
        tracker.update(predictions, targets)
        
        metrics = tracker.compute()
        
        assert 'accuracy' in metrics
        assert 'f1_micro' in metrics
        assert 'f1_macro' in metrics
        assert 'auroc' in metrics
        
        # Reset and test
        tracker.reset()
        assert len(tracker.predictions) == 0
        assert len(tracker.targets) == 0


class TestUtilities:
    """Test utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate some random numbers
        torch_rand = torch.rand(5)
        np_rand = np.random.rand(5)
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.rand(5)
        np_rand2 = np.random.rand(5)
        
        # Should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = torch.nn.Linear(10, 5)
        num_params = count_parameters(model)
        assert num_params == 55  # 10*5 + 5 bias


if __name__ == "__main__":
    pytest.main([__file__])
