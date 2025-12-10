"""Enhanced GraphSAGE model implementation."""

from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data, Batch


class GraphSAGE(nn.Module):
    """GraphSAGE model with multiple aggregation strategies and advanced features.
    
    This implementation supports:
    - Multiple aggregation strategies (mean, max, LSTM)
    - Residual connections
    - Batch normalization
    - Dropout
    - Neighbor sampling for large graphs
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        num_layers: int = 2,
        aggregator: str = "mean",
        dropout: float = 0.5,
        use_batch_norm: bool = True,
        use_residual: bool = False,
        activation: str = "relu",
    ):
        """Initialize GraphSAGE model.
        
        Args:
            in_channels: Number of input features.
            hidden_channels: List of hidden channel sizes.
            out_channels: Number of output classes.
            num_layers: Number of GraphSAGE layers.
            aggregator: Aggregation strategy ('mean', 'max', 'lstm').
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
            use_residual: Whether to use residual connections.
            activation: Activation function ('relu', 'elu', 'gelu').
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.aggregator = aggregator
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.residual_projs = nn.ModuleList() if use_residual else None
        
        # Input layer
        self.convs.append(
            SAGEConv(
                in_channels,
                hidden_channels[0],
                aggr=aggregator,
            )
        )
        
        if use_batch_norm:
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels[0]))
        
        if use_residual and in_channels != hidden_channels[0]:
            self.residual_projs.append(nn.Linear(in_channels, hidden_channels[0]))
        elif use_residual:
            self.residual_projs.append(nn.Identity())
        
        # Hidden layers
        for i in range(1, num_layers - 1):
            self.convs.append(
                SAGEConv(
                    hidden_channels[i - 1],
                    hidden_channels[i],
                    aggr=aggregator,
                )
            )
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_channels[i]))
            
            if use_residual and hidden_channels[i - 1] != hidden_channels[i]:
                self.residual_projs.append(nn.Linear(hidden_channels[i - 1], hidden_channels[i]))
            elif use_residual:
                self.residual_projs.append(nn.Identity())
        
        # Output layer
        if num_layers > 1:
            self.convs.append(
                SAGEConv(
                    hidden_channels[-1],
                    out_channels,
                    aggr=aggregator,
                )
            )
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(out_channels))
            
            if use_residual and hidden_channels[-1] != out_channels:
                self.residual_projs.append(nn.Linear(hidden_channels[-1], out_channels))
            elif use_residual:
                self.residual_projs.append(nn.Identity())
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "elu":
            self.activation = F.elu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            batch: Batch vector for graph-level tasks.
            
        Returns:
            Node embeddings or logits.
        """
        h = x
        
        for i, conv in enumerate(self.convs):
            # Store input for residual connection
            h_in = h
            
            # GraphSAGE convolution
            h = conv(h, edge_index)
            
            # Batch normalization
            if self.batch_norms is not None:
                h = self.batch_norms[i](h)
            
            # Residual connection
            if self.use_residual and i > 0:
                h = h + self.residual_projs[i](h_in)
            
            # Activation (except for last layer)
            if i < len(self.convs) - 1:
                h = self.activation(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h
    
    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get node embeddings (without final classification layer).
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            batch: Batch vector for graph-level tasks.
            
        Returns:
            Node embeddings.
        """
        h = x
        
        for i, conv in enumerate(self.convs[:-1]):  # Exclude last layer
            h_in = h
            h = conv(h, edge_index)
            
            if self.batch_norms is not None:
                h = self.batch_norms[i](h)
            
            if self.use_residual and i > 0:
                h = h + self.residual_projs[i](h_in)
            
            h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h


class GraphSAGEPooling(nn.Module):
    """GraphSAGE with pooling for graph-level tasks."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        out_channels: int,
        num_layers: int = 2,
        aggregator: str = "mean",
        pooling: str = "mean",
        dropout: float = 0.5,
        use_batch_norm: bool = True,
    ):
        """Initialize GraphSAGE with pooling.
        
        Args:
            in_channels: Number of input features.
            hidden_channels: List of hidden channel sizes.
            out_channels: Number of output classes.
            num_layers: Number of GraphSAGE layers.
            aggregator: Aggregation strategy.
            pooling: Pooling strategy ('mean', 'max', 'add', 'attention').
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()
        
        self.graphsage = GraphSAGE(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels[-1],
            num_layers=num_layers,
            aggregator=aggregator,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
        
        self.pooling = pooling
        
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
                nn.Tanh(),
                nn.Linear(hidden_channels[-1] // 2, 1),
            )
        
        self.classifier = nn.Linear(hidden_channels[-1], out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pooling.
        
        Args:
            x: Node features.
            edge_index: Graph connectivity.
            batch: Batch vector.
            
        Returns:
            Graph-level predictions.
        """
        # Get node embeddings
        h = self.graphsage.get_embeddings(x, edge_index, batch)
        
        # Pool to graph level
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        
        if self.pooling == "mean":
            graph_emb = global_mean_pool(h, batch)
        elif self.pooling == "max":
            graph_emb = global_max_pool(h, batch)
        elif self.pooling == "add":
            graph_emb = global_add_pool(h, batch)
        elif self.pooling == "attention":
            att_weights = self.attention(h)
            att_weights = F.softmax(att_weights, dim=0)
            graph_emb = global_add_pool(h * att_weights, batch)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classification
        return self.classifier(graph_emb)
