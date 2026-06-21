"""Xuất báo cáo PDF đầy đủ."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class PDFExporter:
    @staticmethod
    def export(
        output_path: Path,
        image_name: str,
        stats: dict,
        health_report_text: str,
        plain_language: str,
        analyzed_image_path: Optional[Path] = None,
        chart_data: Optional[Dict] = None,
    ) -> bool:
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                Image,
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError:
            return False

        doc = SimpleDocTemplate(str(output_path), pagesize=A4, topMargin=1.5 * cm)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title", parent=styles["Heading1"], fontSize=16, textColor=colors.HexColor("#667eea")
        )
        body = styles["Normal"]
        story = []

        story.append(Paragraph("PANACEA — Báo cáo Phân tích Tế bào Máu", title_style))
        story.append(Paragraph(f"Mẫu: {image_name} | {datetime.now().strftime('%d/%m/%Y %H:%M')}", body))
        story.append(Spacer(1, 0.5 * cm))

        if analyzed_image_path and analyzed_image_path.exists():
            try:
                img = Image(str(analyzed_image_path), width=14 * cm, height=10 * cm)
                story.append(img)
                story.append(Spacer(1, 0.3 * cm))
            except Exception:
                pass

        counts = stats.get("cell_counts", {})
        pcts = stats.get("cell_percentages", {})
        table_data = [["Loại tế bào", "Số lượng", "Tỷ lệ %"]]
        for ct in ("RBC", "WBC", "Platelets"):
            table_data.append([ct, str(counts.get(ct, 0)), f"{pcts.get(ct, 0):.1f}%"])
        table_data.append(["Tổng", str(stats.get("total_cells", 0)), "100%"])
        t = Table(table_data, colWidths=[5 * cm, 4 * cm, 4 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#667eea")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
        ]))
        story.append(Paragraph("<b>Thống kê tế bào</b>", body))
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

        warnings = stats.get("warnings", [])
        if warnings:
            story.append(Paragraph("<b>Cảnh báo</b>", body))
            for w in warnings:
                story.append(Paragraph(f"• {w}", body))

        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("<b>Chẩn đoán sức khỏe máu</b>", body))
        for line in health_report_text.split("\n")[:40]:
            if line.strip():
                story.append(Paragraph(line.replace("<", "&lt;"), body))

        story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("<b>Giải thích dễ hiểu</b>", body))
        for line in plain_language.split("\n")[:25]:
            if line.strip():
                story.append(Paragraph(line.replace("<", "&lt;"), body))

        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(
            "<i>Kết quả chỉ mang tính hỗ trợ — không thay thế chẩn đoán y khoa.</i>",
            body,
        ))

        doc.build(story)
        return True
