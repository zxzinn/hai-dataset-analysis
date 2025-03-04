import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
import polars as pl
from typing import List, Optional, Union, Dict, Any
from pathlib import Path

class Visualizer:
    """Visualization utilities for HAI Security Dataset"""
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots (optional)
        """
        self.save_dir = Path(save_dir) if save_dir else None
        
    def plot_time_series(self,
                        df: pl.LazyFrame,
                        columns: List[str],
                        attack_col: str = "attack",
                        title: str = "Time Series Plot") -> go.Figure:
        """
        Plot time series data with attack regions highlighted
        
        Args:
            df: Input LazyFrame
            columns: Columns to plot
            attack_col: Column indicating attack periods
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Convert to pandas for plotting
        pdf = df.collect().to_pandas()
        
        # Create subplots
        fig = make_subplots(rows=len(columns), cols=1,
                           shared_xaxes=True,
                           subplot_titles=columns)
        
        # Add traces for each column
        for i, col in enumerate(columns, 1):
            # Add time series
            fig.add_trace(
                go.Scatter(x=pdf.index, y=pdf[col], name=col),
                row=i, col=1
            )
            
            # Highlight attack regions
            attack_regions = pdf[pdf[attack_col] == 1].index
            if len(attack_regions) > 0:
                fig.add_trace(
                    go.Scatter(x=attack_regions, y=pdf[col][attack_regions],
                             mode='markers',
                             marker=dict(color='red', size=8),
                             name=f'{col} (Attack)'),
                    row=i, col=1
                )
        
        fig.update_layout(height=300*len(columns), title_text=title, showlegend=True)
        return fig
    
    def plot_correlation_heatmap(self,
                                corr_matrix: np.ndarray,
                                labels: List[str],
                                title: str = "Feature Correlation Heatmap") -> go.Figure:
        """
        Plot correlation matrix heatmap
        
        Args:
            corr_matrix: Correlation matrix
            labels: Feature labels
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Features",
            yaxis_title="Features",
            width=800,
            height=800
        )
        
        return fig
    
    def plot_attack_distribution(self,
                               df: pl.LazyFrame,
                               attack_cols: List[str],
                               title: str = "Attack Distribution") -> go.Figure:
        """
        Plot distribution of attacks across different processes
        
        Args:
            df: Input LazyFrame
            attack_cols: Columns indicating attacks
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Calculate attack statistics
        attack_stats = df.select([
            pl.col(col).sum().alias(f"{col}_count") for col in attack_cols
        ]).collect()
        
        # Create bar plot
        fig = go.Figure(data=[
            go.Bar(x=attack_cols,
                  y=attack_stats.to_numpy().flatten(),
                  text=attack_stats.to_numpy().flatten(),
                  textposition='auto')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Attack Type",
            yaxis_title="Count",
            width=800,
            height=500
        )
        
        return fig
    
    def plot_confusion_matrix(self,
                            cm: np.ndarray,
                            labels: List[str],
                            title: str = "Confusion Matrix") -> go.Figure:
        """
        Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale='Viridis',
            showscale=True
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Predicted",
            yaxis_title="Actual",
            width=600,
            height=600
        )
        
        return fig
    
    def plot_roc_curve(self,
                      fpr: np.ndarray,
                      tpr: np.ndarray,
                      auc: float,
                      title: str = "ROC Curve") -> go.Figure:
        """
        Plot ROC curve
        
        Args:
            fpr: False positive rates
            tpr: True positive rates
            auc: Area under curve
            title: Plot title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Add ROC curve
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr,
                      name=f'ROC (AUC = {auc:.3f})',
                      mode='lines')
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1],
                      name='Random',
                      mode='lines',
                      line=dict(dash='dash'))
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            width=600,
            height=600
        )
        
        return fig
    
    def plot_attack_propagation(self,
                              G: nx.Graph,
                              attack_path: List[str],
                              title: str = "Attack Propagation Path") -> go.Figure:
        """
        Plot attack propagation path in network
        
        Args:
            G: NetworkX graph
            attack_path: List of nodes in attack path
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Create layout
        pos = nx.spring_layout(G)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_x = []
        node_y = []
        node_colors = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_colors.append('red' if node in attack_path else 'lightblue')
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                color=node_colors,
                size=10,
                line_width=2))
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           width=800,
                           height=800
                       ))
        
        return fig
    
    def save_figure(self, fig: go.Figure, filename: str) -> None:
        """
        Save figure to file
        
        Args:
            fig: Plotly figure to save
            filename: Output filename
        """
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            fig.write_html(self.save_dir / f"{filename}.html")
            fig.write_image(self.save_dir / f"{filename}.png")
