"""
Analytics utilities for object detection results.
"""
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
from collections import Counter

from ..models.detection_models import DetectionResult


class AnalyticsProcessor:
    """Utility class for processing detection analytics."""
    
    def __init__(self):
        """Initialize analytics processor."""
        self.logger = logging.getLogger(__name__)
    
    def generate_detection_summary(self, results: List[DetectionResult]) -> Dict[str, Any]:
        """Generate summary statistics for detection results."""
        try:
            if not results:
                return {}
            
            summary = {
                'total_services': len(results),
                'total_detections': sum(len(r.detections) for r in results),
                'average_processing_time': sum(r.processing_time for r in results) / len(results),
                'services_used': [r.service_name for r in results],
                'detection_counts_by_service': {r.service_name: len(r.detections) for r in results}
            }
            
            # Get all detections across services
            all_detections = []
            for result in results:
                all_detections.extend(result.detections)
            
            if all_detections:
                # Calculate confidence statistics
                confidences = [d.confidence for d in all_detections]
                summary.update({
                    'average_confidence': sum(confidences) / len(confidences),
                    'max_confidence': max(confidences),
                    'min_confidence': min(confidences),
                    'high_confidence_count': len([c for c in confidences if c >= 0.7]),
                    'medium_confidence_count': len([c for c in confidences if 0.4 <= c < 0.7]),
                    'low_confidence_count': len([c for c in confidences if c < 0.4])
                })
                
                # Get tag statistics
                tag_counts = Counter(d.tag_name for d in all_detections)
                summary['most_common_tags'] = tag_counts.most_common(5)
                summary['unique_tags'] = len(tag_counts)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate detection summary: {e}")
            return {}
    
    def create_confidence_distribution_chart(self, results: List[DetectionResult]) -> go.Figure:
        """Create a histogram showing confidence distribution."""
        try:
            all_confidences = []
            service_labels = []
            
            for result in results:
                confidences = [d.confidence for d in result.detections]
                all_confidences.extend(confidences)
                service_labels.extend([result.service_name] * len(confidences))
            
            if not all_confidences:
                # Return empty figure
                fig = go.Figure()
                fig.add_annotation(
                    text="No detections to display",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            df = pd.DataFrame({
                'confidence': all_confidences,
                'service': service_labels
            })
            
            fig = px.histogram(
                df, 
                x='confidence', 
                color='service',
                title='Detection Confidence Distribution',
                labels={'confidence': 'Confidence Score', 'count': 'Number of Detections'},
                nbins=20
            )
            
            fig.update_layout(
                xaxis_title="Confidence Score",
                yaxis_title="Number of Detections",
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create confidence distribution chart: {e}")
            return go.Figure()
    
    def create_detection_comparison_chart(self, results: List[DetectionResult]) -> go.Figure:
        """Create a bar chart comparing detections across services."""
        try:
            if not results:
                fig = go.Figure()
                fig.add_annotation(
                    text="No results to compare",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            services = [r.service_name for r in results]
            detection_counts = [len(r.detections) for r in results]
            processing_times = [r.processing_time for r in results]
            
            # Create subplots
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Detection Count by Service', 'Processing Time by Service'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Add detection count bar chart
            fig.add_trace(
                go.Bar(x=services, y=detection_counts, name='Detection Count', marker_color='skyblue'),
                row=1, col=1
            )
            
            # Add processing time bar chart
            fig.add_trace(
                go.Bar(x=services, y=processing_times, name='Processing Time (s)', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text="Service Performance Comparison",
                showlegend=False,
                height=400
            )
            
            fig.update_xaxes(title_text="Service", row=1, col=1)
            fig.update_xaxes(title_text="Service", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_yaxes(title_text="Time (seconds)", row=1, col=2)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create comparison chart: {e}")
            return go.Figure()
    
    def create_tag_distribution_chart(self, results: List[DetectionResult]) -> go.Figure:
        """Create a pie chart showing distribution of detected object tags."""
        try:
            all_tags = []
            for result in results:
                all_tags.extend([d.tag_name for d in result.detections])
            
            if not all_tags:
                fig = go.Figure()
                fig.add_annotation(
                    text="No tags to display",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, xanchor='center', yanchor='middle',
                    showarrow=False, font=dict(size=16)
                )
                return fig
            
            tag_counts = Counter(all_tags)
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(tag_counts.keys()),
                    values=list(tag_counts.values()),
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="Distribution of Detected Object Types",
                annotations=[dict(text='Objects', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Failed to create tag distribution chart: {e}")
            return go.Figure()
    
    def export_results_to_dataframe(self, results: List[DetectionResult]) -> pd.DataFrame:
        """Export detection results to a pandas DataFrame."""
        try:
            data = []
            
            for result in results:
                for detection in result.detections:
                    data.append({
                        'service': result.service_name,
                        'tag_name': detection.tag_name,
                        'confidence': detection.confidence,
                        'bbox_left': detection.bounding_box.left,
                        'bbox_top': detection.bounding_box.top,
                        'bbox_width': detection.bounding_box.width,
                        'bbox_height': detection.bounding_box.height,
                        'processing_time': result.processing_time
                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Failed to export results to DataFrame: {e}")
            return pd.DataFrame()
