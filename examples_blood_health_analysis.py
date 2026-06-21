#!/usr/bin/env python3
"""
Ví dụ sử dụng: Phân tích Sức khỏe Máu (Blood Health Analysis)

Tập lệnh này minh họa cách sử dụng lớp BloodHealthAnalyzer để:
1. Phân tích các tế bào phát hiện được
2. So sánh với giá trị bình thường
3. Đưa ra kết luận về tình trạng máu
"""

import sys
from pathlib import Path

# Đảm bảo src có thể import được
sys.path.insert(0, str(Path(__file__).parent))

from src.diagnosis.blood_health_analyzer import BloodHealthAnalyzer, BloodHealthStatus


def example_1_normal_blood():
    """Ví dụ 1: Máu bình thường"""
    print("\n" + "="*70)
    print("VÍ DỤ 1: Phân tích máu bình thường")
    print("="*70 + "\n")
    
    # Dữ liệu tế bào bình thường
    cell_counts = {
        "RBC": 450,      # 4.5 triệu/μL (bình thường)
        "WBC": 50,       # 5 nghìn/μL (bình thường)
        "Platelets": 200 # 200 nghìn/μL (bình thường)
    }
    
    cell_percentages = {
        "RBC": 85.7,
        "WBC": 9.5,
        "Platelets": 38.1
    }
    
    total_cells = 524
    
    # Phân tích
    analyzer = BloodHealthAnalyzer()
    report = analyzer.analyze(cell_counts, cell_percentages, total_cells)
    
    # Hiển thị kết quả
    print(report.to_text())


def example_2_anemia():
    """Ví dụ 2: Thiếu máu (Anemia)"""
    print("\n" + "="*70)
    print("VÍ DỤ 2: Phân tích thiếu máu")
    print("="*70 + "\n")
    
    # Dữ liệu tế bào bệnh nhân thiếu máu
    cell_counts = {
        "RBC": 250,      # 2.5 triệu/μL (thấp - thiếu máu)
        "WBC": 45,       # 4.5 nghìn/μL (bình thường)
        "Platelets": 180 # 180 nghìn/μL (bình thường)
    }
    
    cell_percentages = {
        "RBC": 75.0,
        "WBC": 13.5,
        "Platelets": 54.1
    }
    
    total_cells = 334
    
    # Phân tích
    analyzer = BloodHealthAnalyzer()
    report = analyzer.analyze(cell_counts, cell_percentages, total_cells)
    
    # Hiển thị kết quả
    print(report.to_text())


def example_3_leukocytosis():
    """Ví dụ 3: Tăng bạch cầu (Leukocytosis)"""
    print("\n" + "="*70)
    print("VÍ DỤ 3: Phân tích tăng bạch cầu")
    print("="*70 + "\n")
    
    # Dữ liệu tế bào bệnh nhân tăng bạch cầu
    cell_counts = {
        "RBC": 430,      # 4.3 triệu/μL (bình thường)
        "WBC": 150,      # 15 nghìn/μL (cao - tăng bạch cầu)
        "Platelets": 210 # 210 nghìn/μL (bình thường)
    }
    
    cell_percentages = {
        "RBC": 68.3,
        "WBC": 23.8,
        "Platelets": 33.3
    }
    
    total_cells = 630
    
    # Phân tích
    analyzer = BloodHealthAnalyzer()
    report = analyzer.analyze(cell_counts, cell_percentages, total_cells)
    
    # Hiển thị kết quả
    print(report.to_text())


def example_4_thrombocytopenia():
    """Ví dụ 4: Giảm tiểu cầu (Thrombocytopenia)"""
    print("\n" + "="*70)
    print("VÍ DỤ 4: Phân tích giảm tiểu cầu")
    print("="*70 + "\n")
    
    # Dữ liệu tế bào bệnh nhân giảm tiểu cầu
    cell_counts = {
        "RBC": 470,    # 4.7 triệu/μL (bình thường)
        "WBC": 55,     # 5.5 nghìn/μL (bình thường)
        "Platelets": 80  # 80 nghìn/μL (thấp - giảm tiểu cầu)
    }
    
    cell_percentages = {
        "RBC": 85.5,
        "WBC": 10.0,
        "Platelets": 14.5
    }
    
    total_cells = 550
    
    # Phân tích
    analyzer = BloodHealthAnalyzer()
    report = analyzer.analyze(cell_counts, cell_percentages, total_cells)
    
    # Hiển thị kết quả
    print(report.to_text())


def example_5_critical_state():
    """Ví dụ 5: Tình trạng nguy hiểm (No RBC detected)"""
    print("\n" + "="*70)
    print("VÍ DỤ 5: Phân tích tình trạng nguy hiểm")
    print("="*70 + "\n")
    
    # Dữ liệu: không phát hiện được hồng cầu
    cell_counts = {
        "RBC": 0,        # Không phát hiện
        "WBC": 40,       # 4 nghìn/μL
        "Platelets": 150 # 150 nghìn/μL
    }
    
    cell_percentages = {
        "RBC": 0.0,
        "WBC": 21.0,
        "Platelets": 79.0
    }
    
    total_cells = 190
    
    # Phân tích
    analyzer = BloodHealthAnalyzer()
    report = analyzer.analyze(cell_counts, cell_percentages, total_cells)
    
    # Hiển thị kết quả
    print(report.to_text())


if __name__ == "__main__":
    # Chạy tất cả các ví dụ
    example_1_normal_blood()
    example_2_anemia()
    example_3_leukocytosis()
    example_4_thrombocytopenia()
    example_5_critical_state()
    
    print("\n" + "="*70)
    print("Hoàn tất các ví dụ!")
    print("="*70)
