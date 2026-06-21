"""Giải thích báo cáo bằng ngôn ngữ dễ hiểu — local hoặc qua OpenAI API."""

import os
from typing import Optional


class ReportExplainer:
    """Chuyển báo cáo kỹ thuật sang ngôn ngữ dễ hiểu."""

    def explain(
        self,
        report_text: str,
        health_summary: str,
        cell_counts: dict,
        cell_percentages: dict,
    ) -> str:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key:
            result = self._explain_via_openai(report_text, health_summary, api_key)
            if result:
                return result
        return self._explain_local(cell_counts, cell_percentages, health_summary)

    def _explain_local(
        self,
        cell_counts: dict,
        cell_percentages: dict,
        health_summary: str,
    ) -> str:
        rbc = cell_counts.get("RBC", 0)
        wbc = cell_counts.get("WBC", 0)
        plt = cell_counts.get("Platelets", 0)
        total = sum(cell_counts.values()) or 1

        lines = [
            "GIẢI THÍCH DỄ HIỂU",
            "=" * 50,
            "",
            f"Hệ thống đã phân tích ảnh lam máu và tìm thấy {total} tế bào:",
            f"  • {rbc} hồng cầu (RBC) — chiếm {cell_percentages.get('RBC', 0):.1f}%",
            f"  • {wbc} bạch cầu (WBC) — chiếm {cell_percentages.get('WBC', 0):.1f}%",
            f"  • {plt} tiểu cầu — chiếm {cell_percentages.get('Platelets', 0):.1f}%",
            "",
            "Hồng cầu vận chuyển oxy, bạch cầu chống nhiễm trùng, tiểu cầu giúp đông máu.",
            "",
            f"Kết luận sơ bộ: {health_summary}",
            "",
            "Lưu ý quan trọng:",
            "  • Đây là công cụ hỗ trợ AI, không thay thế bác sĩ.",
            "  • Cần xét nghiệm máu (CBC) tại phòng lab để xác nhận.",
            "  • Chất lượng ảnh và góc chụp ảnh hưởng đến độ chính xác.",
        ]
        return "\n".join(lines)

    def _explain_via_openai(
        self, report_text: str, health_summary: str, api_key: str
    ) -> Optional[str]:
        try:
            import urllib.request
            import json

            prompt = (
                "Bạn là trợ lý y tế. Giải thích kết quả phân tích tế bào máu sau "
                "bằng tiếng Việt, ngôn ngữ dễ hiểu cho người không chuyên, "
                "kèm lưu ý không thay thế bác sĩ:\n\n"
                f"{health_summary}\n\n{report_text[:2000]}"
            )
            payload = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 600,
            }).encode()
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
        except Exception:
            return None
