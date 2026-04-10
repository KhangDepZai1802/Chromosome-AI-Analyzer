# src/core/exporter.py
import csv
import os
import shutil
import cv2
from datetime import datetime


class Exporter:
    @staticmethod
    def export_to_csv(file_path, image_path, count_value, status_plain):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        image_name = os.path.basename(image_path) if image_path else ""

        with open(file_path, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["Thời gian phân tích", "Tên file ảnh", "Số lượng NST", "Kết luận / đánh giá"]
            )
            writer.writerow([current_time, image_name, count_value, status_plain])

    @staticmethod
    def save_result_bundle(
        folder_path,
        source_image_path,
        result_bgr_image,
        count_value,
        status_plain,
        normal_count,
        tolerance,
    ):
        """
        Lưu ảnh gốc (bản sao), ảnh kết quả phân tích, và file báo cáo .txt
        """
        os.makedirs(folder_path, exist_ok=True)
        base = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = os.path.splitext(os.path.basename(source_image_path or "anh"))[0]

        report_name = f"bao_cao_{stem}_{base}.txt"
        report_path = os.path.join(folder_path, report_name)

        if source_image_path and os.path.isfile(source_image_path):
            ext = os.path.splitext(source_image_path)[1] or ".png"
            orig_dst = os.path.join(folder_path, f"anh_goc_{stem}_{base}{ext}")
            shutil.copy2(source_image_path, orig_dst)

        seg_name = f"anh_phan_doan_{stem}_{base}.png"
        seg_path = os.path.join(folder_path, seg_name)
        if result_bgr_image is not None:
            cv2.imwrite(seg_path, result_bgr_image)

        lines = [
            "BÁO CÁO PHÂN TÍCH NST (tự động)",
            "=" * 40,
            f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Ảnh gốc: {os.path.basename(source_image_path) if source_image_path else '(không có)'}",
            f"Ngưỡng chuẩn: {normal_count} ± {tolerance}",
            "",
            f"Số lượng NST đếm được: {count_value}",
            "",
            status_plain,
            "",
            "-" * 40,
            "Lưu ý: Kết quả hỗ trợ chẩn đoán, không thay thế xét nghiệm chuyên sâu.",
        ]
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path, seg_path if result_bgr_image is not None else None
