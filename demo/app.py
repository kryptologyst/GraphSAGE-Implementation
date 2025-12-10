"""Interactive Streamlit demo for GraphSAGE."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network
import tempfile
import os

from src.models import GraphSAGE
from src.data import load_dataset, generate_synthetic_graph, get_graph_statistics
from src.eval import evaluate_model_performance, visualize_embeddings
from src.utils import set_seed, get_device


def load_model_and_data(dataset_name: str, model_path: str = None):
    """Load model and data for the demo."""
    try:
        # Load data
        if dataset_name == "synthetic":
            data = generate_synthetic_graph(num_nodes=500, num_classes=7, num_features=1433)
        else:
            data, _ = load_dataset(dataset_name)
        
        # Load model if available
        model = None
        if model_path and os.path.exists(model_path):
            device = get_device()
            model = GraphSAGE(
                in_channels=data.num_node_features,
                hidden_channels=[64, 32],
                out_channels=data.num_classes,
            ).to(device)
            
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        
        return model, data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


def create_network_visualization(data, selected_nodes=None, max_nodes=100):
    """Create interactive network visualization."""
    # Sample nodes if graph is too large
    if data.num_nodes > max_nodes:
        if selected_nodes is None:
            node_indices = torch.randperm(data.num_nodes)[:max_nodes]
        else:
            # Include selected nodes and their neighbors
            selected_set = set(selected_nodes)
            neighbors = set()
            for i in selected_nodes:
                neighbors.update(data.edge_index[1][data.edge_index[0] == i].tolist())
            all_nodes = selected_set.union(neighbors)
            if len(all_nodes) > max_nodes:
                all_nodes = list(all_nodes)[:max_nodes]
            node_indices = torch.tensor(list(all_nodes))
    else:
        node_indices = torch.arange(data.num_nodes)
    
    # Create subgraph
    subgraph = data.subgraph(node_indices)
    
    # Convert to NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(len(node_indices)))
    G.add_edges_from(subgraph.edge_index.t().tolist())
    
    # Create pyvis network
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Add nodes
    for i, node_id in enumerate(node_indices):
        node_id = node_id.item()
        label = f"Node {node_id}"
        color = "#ff6b6b" if selected_nodes and node_id in selected_nodes else "#4ecdc4"
        
        # Add class information if available
        if hasattr(subgraph, 'y'):
            class_id = subgraph.y[i].item()
            label += f" (Class {class_id})"
        
        net.add_node(node_id, label=label, color=color, size=20)
    
    # Add edges
    for edge in subgraph.edge_index.t():
        net.add_edge(edge[0].item(), edge[1].item())
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    return net


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="GraphSAGE Demo",
        page_icon="🕸️",
        layout="wide",
    )
    
    st.title("🕸️ GraphSAGE Interactive Demo")
    st.markdown("Explore GraphSAGE model predictions and graph structure interactively.")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_options = {
        "Cora": "cora",
        "Citeseer": "citeseer", 
        "Pubmed": "pubmed",
        "Synthetic": "synthetic",
    }
    
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        options=list(dataset_options.keys()),
        index=0,
    )
    
    dataset_name = dataset_options[selected_dataset]
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path (optional)",
        value="checkpoints/best_model.pt",
        help="Path to trained model checkpoint",
    )
    
    # Load data and model
    with st.spinner("Loading data and model..."):
        model, data = load_model_and_data(dataset_name, model_path)
    
    if data is None:
        st.error("Failed to load data. Please check your configuration.")
        return
    
    # Display dataset statistics
    st.subheader("📊 Dataset Statistics")
    stats = get_graph_statistics(data)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodes", stats['num_nodes'])
    with col2:
        st.metric("Edges", stats['num_edges'])
    with col3:
        st.metric("Features", stats['num_features'])
    with col4:
        st.metric("Classes", stats['num_classes'])
    
    # Class distribution
    st.subheader("📈 Class Distribution")
    class_df = pd.DataFrame({
        'Class': range(stats['num_classes']),
        'Proportion': stats['class_distribution']
    })
    
    fig = px.bar(class_df, x='Class', y='Proportion', title='Class Distribution')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model predictions (if model is available)
    if model is not None:
        st.subheader("🎯 Model Predictions")
        
        device = get_device()
        data_device = data.to(device)
        
        with torch.no_grad():
            logits = model(data_device.x, data_device.edge_index)
            predictions = logits.argmax(dim=1)
            probabilities = torch.softmax(logits, dim=1)
        
        # Overall performance
        results = evaluate_model_performance(model, data_device, device)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Test Accuracy", f"{results.get('test', {}).get('accuracy', 0):.3f}")
        with col2:
            st.metric("Test F1 Micro", f"{results.get('test', {}).get('f1_micro', 0):.3f}")
        with col3:
            st.metric("Test F1 Macro", f"{results.get('test', {}).get('f1_macro', 0):.3f}")
        
        # Node exploration
        st.subheader("🔍 Node Exploration")
        
        # Node selection
        node_id = st.number_input(
            "Select Node ID",
            min_value=0,
            max_value=data.num_nodes - 1,
            value=0,
            step=1,
        )
        
        if st.button("Analyze Node"):
            # Get node information
            node_features = data.x[node_id]
            node_label = data.y[node_id].item()
            node_prediction = predictions[node_id].item()
            node_probabilities = probabilities[node_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Node Information:**")
                st.write(f"- Node ID: {node_id}")
                st.write(f"- True Class: {node_label}")
                st.write(f"- Predicted Class: {node_prediction}")
                st.write(f"- Prediction Correct: {'✅' if node_label == node_prediction else '❌'}")
                
                # Confidence
                max_prob = node_probabilities.max().item()
                st.write(f"- Confidence: {max_prob:.3f}")
            
            with col2:
                st.write("**Class Probabilities:**")
                prob_df = pd.DataFrame({
                    'Class': range(data.num_classes),
                    'Probability': node_probabilities.cpu().numpy()
                })
                
                fig = px.bar(prob_df, x='Class', y='Probability', title=f'Node {node_id} Class Probabilities')
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature analysis
            st.write("**Top Features:**")
            feature_values = node_features.cpu().numpy()
            top_features = np.argsort(np.abs(feature_values))[-10:][::-1]
            
            feature_df = pd.DataFrame({
                'Feature': top_features,
                'Value': feature_values[top_features]
            })
            
            fig = px.bar(feature_df, x='Feature', y='Value', title=f'Top Features for Node {node_id}')
            st.plotly_chart(fig, use_container_width=True)
            
            # Neighbor analysis
            st.write("**Neighbor Analysis:**")
            neighbors = data.edge_index[1][data.edge_index[0] == node_id]
            neighbor_labels = data.y[neighbors]
            neighbor_predictions = predictions[neighbors]
            
            neighbor_df = pd.DataFrame({
                'Neighbor': neighbors.cpu().numpy(),
                'True Class': neighbor_labels.cpu().numpy(),
                'Predicted Class': neighbor_predictions.cpu().numpy(),
            })
            
            st.dataframe(neighbor_df, use_container_width=True)
    
    # Graph visualization
    st.subheader("🕸️ Graph Visualization")
    
    # Visualization options
    col1, col2 = st.columns(2)
    with col1:
        max_nodes = st.slider("Max Nodes to Display", 50, 500, 100)
    with col2:
        selected_nodes = st.multiselect(
            "Highlight Nodes",
            options=list(range(min(100, data.num_nodes))),
            default=[],
        )
    
    if st.button("Generate Visualization"):
        with st.spinner("Generating graph visualization..."):
            net = create_network_visualization(data, selected_nodes, max_nodes)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
                net.save_graph(tmp_file.name)
                
                # Read and display
                with open(tmp_file.name, 'r') as f:
                    html_content = f.read()
                
                st.components.v1.html(html_content, height=600)
                
                # Clean up
                os.unlink(tmp_file.name)
    
    # Embedding visualization
    if model is not None:
        st.subheader("🎨 Embedding Visualization")
        
        if st.button("Generate Embedding Plot"):
            with st.spinner("Generating embedding visualization..."):
                try:
                    # Get embeddings
                    device = get_device()
                    data_device = data.to(device)
                    
                    with torch.no_grad():
                        embeddings = model.get_embeddings(data_device.x, data_device.edge_index)
                        embeddings = embeddings.cpu().numpy()
                        labels = data.y.cpu().numpy()
                    
                    # Create t-SNE plot
                    from sklearn.manifold import TSNE
                    tsne = TSNE(n_components=2, random_state=42)
                    embeddings_2d = tsne.fit_transform(embeddings)
                    
                    # Create plot
                    fig = px.scatter(
                        x=embeddings_2d[:, 0],
                        y=embeddings_2d[:, 1],
                        color=labels,
                        title="Node Embeddings (t-SNE)",
                        labels={'x': 't-SNE 1', 'y': 't-SNE 2'},
                        color_continuous_scale='viridis'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error generating embedding plot: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and PyTorch Geometric")


if __name__ == "__main__":
    main()
