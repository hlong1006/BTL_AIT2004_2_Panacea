from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any
from collections import Counter

import pandas as pd
import json


@dataclass
class CellStatistics:
    """Store aggregated cell statistics for a blood sample analysis."""
    
    total_cells: int = 0
    cell_counts: Dict[str, int] = field(default_factory=dict)
    cell_percentages: Dict[str, float] = field(default_factory=dict)
    feature_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message to the statistics."""
        if warning not in self.warnings:
            self.warnings.append(warning)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert statistics to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert statistics summary to DataFrame."""
        data = []
        for cell_type, count in self.cell_counts.items():
            percentage = self.cell_percentages.get(cell_type, 0.0)
            data.append({
                "Cell Type": cell_type,
                "Count": count,
                "Percentage": f"{percentage:.2f}%"
            })
        return pd.DataFrame(data)


class StatisticsCalculator:
    """Calculate statistics from cell predictions."""
    
    # Normal ranges for blood cell counts (reference values)
    NORMAL_RANGES = {
        "RBC": {"min": 4.5, "max": 5.5, "unit": "million/μL"},      # Red blood cells
        "WBC": {"min": 4.5, "max": 11.0, "unit": "thousand/μL"},    # White blood cells
        "Platelets": {"min": 150, "max": 400, "unit": "thousand/μL"}, # Platelets
    }
    
    @staticmethod
    def calculate_statistics(
        labels: List[str],
        features_df: pd.DataFrame
    ) -> CellStatistics:
        """Calculate comprehensive statistics from predictions."""
        
        stats = CellStatistics()
        stats.total_cells = len(labels)
        
        # Count cells by type
        counter = Counter(labels)
        stats.cell_counts = dict(counter)
        
        # Calculate percentages
        if stats.total_cells > 0:
            for cell_type, count in stats.cell_counts.items():
                percentage = (count / stats.total_cells) * 100
                stats.cell_percentages[cell_type] = percentage
        
        # Calculate feature statistics for each cell type
        if features_df is not None and len(features_df) > 0:
            features_df["predicted_label"] = labels
            for cell_type in stats.cell_counts.keys():
                mask = features_df["predicted_label"] == cell_type
                cell_features = features_df[mask]
                
                if len(cell_features) > 0:
                    stats.feature_stats[cell_type] = {
                        "avg_area": float(cell_features.get("area", pd.Series(0)).mean()),
                        "avg_perimeter": float(cell_features.get("perimeter", pd.Series(0)).mean()),
                        "avg_circularity": float(cell_features.get("circularity", pd.Series(0)).mean()),
                        "avg_texture": float(cell_features.get("texture_laplacian_var", pd.Series(0)).mean()),
                    }
        
        # Generate warnings for abnormal values
        StatisticsCalculator._generate_warnings(stats)
        
        return stats
    
    @staticmethod
    def _generate_warnings(stats: CellStatistics) -> None:
        """Generate warnings for abnormal cell counts."""
        
        total = stats.total_cells
        
        # Check for low RBC count (anemia)
        rbc_count = stats.cell_counts.get("RBC", 0)
        if total > 0:
            rbc_percentage = (rbc_count / total) * 100
            if rbc_percentage < 40:
                stats.add_warning(
                    f"⚠️ RBC low: {rbc_count} cells ({rbc_percentage:.1f}%) - "
                    "Could indicate anemia or blood disorder"
                )
        
        # Check for high WBC count (leukocytosis)
        wbc_count = stats.cell_counts.get("WBC", 0)
        if total > 0:
            wbc_percentage = (wbc_count / total) * 100
            if wbc_percentage > 20:
                stats.add_warning(
                    f"⚠️ WBC high: {wbc_count} cells ({wbc_percentage:.1f}%) - "
                    "Could indicate infection or immune disorder"
                )
        
        # Check for low WBC count (leukopenia)
        if total > 0 and wbc_percentage < 2:
            stats.add_warning(
                f"⚠️ WBC critically low: {wbc_count} cells ({wbc_percentage:.1f}%) - "
                "Could indicate severe immunosuppression"
            )
        
        # Check for abnormal platelet count
        platelet_count = stats.cell_counts.get("Platelets", 0)
        if total > 0:
            platelet_percentage = (platelet_count / total) * 100
            if platelet_percentage < 5:
                stats.add_warning(
                    f"⚠️ Platelets critically low: {platelet_count} cells ({platelet_percentage:.1f}%) - "
                    "Could indicate thrombocytopenia"
                )
            elif platelet_percentage > 25:
                stats.add_warning(
                    f"⚠️ Platelets high: {platelet_count} cells ({platelet_percentage:.1f}%) - "
                    "Could indicate thrombocytosis"
                )
        
        # Warn if very few cells detected
        if total < 5:
            stats.add_warning(
                f"⚠️ Very few cells detected: {total} - "
                "May need to re-scan or adjust microscope"
            )
        
        # Warn if extremely many cells (possible segmentation error)
        if total > 500:
            stats.add_warning(
                f"⚠️ Very high cell count: {total} - "
                "May indicate over-segmentation or aggregated cells"
            )
    
    @staticmethod
    def generate_report_text(stats: CellStatistics, image_name: str = "Blood Sample") -> str:
        """Generate a text report."""
        
        report = []
        report.append("=" * 60)
        report.append("BLOOD CELL ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"\nSample: {image_name}")
        report.append(f"\nTotal Cells Detected: {stats.total_cells}")
        report.append("\n" + "-" * 60)
        report.append("CELL COUNT SUMMARY")
        report.append("-" * 60)
        
        # Sort by count (descending)
        sorted_cells = sorted(stats.cell_counts.items(), key=lambda x: x[1], reverse=True)
        
        for cell_type, count in sorted_cells:
            percentage = stats.cell_percentages.get(cell_type, 0.0)
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_length + "░" * (50 - bar_length)
            report.append(f"\n{cell_type:12} | {count:4d} | {percentage:6.2f}% | {bar}")
        
        # Feature statistics
        if stats.feature_stats:
            report.append("\n" + "-" * 60)
            report.append("MORPHOLOGICAL FEATURES (Average per Cell Type)")
            report.append("-" * 60)
            
            for cell_type, features in stats.feature_stats.items():
                report.append(f"\n{cell_type}:")
                report.append(f"  • Average Area:       {features.get('avg_area', 0):.2f} px²")
                report.append(f"  • Average Perimeter:  {features.get('avg_perimeter', 0):.2f} px")
                report.append(f"  • Average Circularity: {features.get('avg_circularity', 0):.4f}")
                report.append(f"  • Average Texture:    {features.get('avg_texture', 0):.4f}")
        
        # Warnings
        if stats.warnings:
            report.append("\n" + "!" * 60)
            report.append("CLINICAL ALERTS")
            report.append("!" * 60)
            for i, warning in enumerate(stats.warnings, 1):
                report.append(f"\n{i}. {warning}")
        else:
            report.append("\n" + "✓" * 60)
            report.append("No abnormal findings detected.")
            report.append("✓" * 60)
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


class ReportExporter:
    """Export statistics to various file formats."""
    
    @staticmethod
    def to_csv(stats: CellStatistics, output_path: str) -> None:
        """Export statistics to CSV."""
        df = stats.to_dataframe()
        df.to_csv(output_path, index=False)
    
    @staticmethod
    def to_json(stats: CellStatistics, output_path: str) -> None:
        """Export statistics to JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(stats.to_json(indent=2))
    
    @staticmethod
    def to_txt(report_text: str, output_path: str) -> None:
        """Export text report to file."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
    
    @staticmethod
    def to_excel(stats: CellStatistics, features_df: pd.DataFrame, output_path: str) -> None:
        """Export to Excel with multiple sheets."""
        try:
            with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
                # Summary sheet
                summary_df = stats.to_dataframe()
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
                # Details sheet
                if features_df is not None and len(features_df) > 0:
                    features_df.to_excel(writer, sheet_name="Cell Details", index=False)
                
                # Warnings sheet
                if stats.warnings:
                    warnings_df = pd.DataFrame({
                        "Alert": stats.warnings
                    })
                    warnings_df.to_excel(writer, sheet_name="Warnings", index=False)
        except ImportError:
            print("⚠️ openpyxl not installed. Skipping Excel export.")
