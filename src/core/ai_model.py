# src/core/ai_model.py
import os
from ultralytics import YOLO
from src.core.chromosome_classifier import ChromosomeClassifier, AnalysisReport


class ChromosomeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_overlap = None
        self.model_anomaly = None
        self.classifier = ChromosomeClassifier()
        self.load_model()

    def _try_load_yolo(self, path: str):
        if not path or not str(path).strip():
            return None
        path = str(path).strip()
        if not os.path.isfile(path):
            print(f"⚠️ Không tìm thấy file mô hình: {path}")
            return None
        ext = os.path.splitext(path)[1].lower()
        if ext not in (".pt", ".pth", ".onnx", ".engine"):
            print(f"⚠️ Định dạng {ext} chưa được tích hợp: {path}")
            return None
        try:
            return YOLO(path)
        except Exception as e:
            print(f"❌ Lỗi load mô hình {path}: {e}")
            return None

    def load_model(self):
        self.model = self._try_load_yolo(self.config.get("model_path", "models/best.pt"))
        self.model_overlap = self._try_load_yolo(self.config.get("model_overlap_path", ""))
        self.model_anomaly = self._try_load_yolo(self.config.get("model_anomaly_path", ""))
        if self.model:
            print("✅ Mô hình phân đoạn/đếm NST đã sẵn sàng.")
        else:
            print("❌ Chưa load được mô hình chính (model_path).")

    def analyze(self, image_path: str):
        """
        Trả về:
            annotated_img   : ảnh BGR đã vẽ kết quả
            total_count     : số NST đếm được
            report_html     : chuỗi HTML hiển thị trong UI
            report_plain    : chuỗi thuần túy để xuất file
            analysis_report : AnalysisReport đầy đủ (dùng cho UI nâng cao)
        """
        if self.model is None:
            raise Exception(
                "Model chưa được load! Kiểm tra đường dẫn trong config.yaml hoặc Cài đặt."
            )

        normal = int(self.config.get("normal_count", 46))
        tolerance = int(self.config.get("tolerance", 1))

        results = self.model.predict(source=image_path, conf=0.25, save=False)
        result = results[0]

        annotated_img = result.plot()
        total_count = len(result.boxes)

        # ── Phân tích chuyên sâu ──────────────────────────────────────────
        report: AnalysisReport = self.classifier.analyze(result, normal, tolerance)

        # ── Tạo báo cáo HTML ─────────────────────────────────────────────
        report_html, report_plain = self._build_reports(report, total_count, normal, tolerance)

        return annotated_img, total_count, report_html, report_plain, report

    # ------------------------------------------------------------------
    # Tạo báo cáo
    # ------------------------------------------------------------------

    def _build_reports(
        self,
        r: AnalysisReport,
        total: int,
        normal: int,
        tolerance: int,
    ):
        diff = abs(total - normal)
        count_color = "#27AE60" if r.is_normal_count else "#C0392B"
        count_label = "Bình thường" if r.is_normal_count else (
            f"Bất thường — {'thừa' if total > normal else 'thiếu'} {diff} NST"
        )

        # Màu mức độ rủi ro
        risk_colors = {
            "Bình thường": "#27AE60",
            "Cần theo dõi": "#E67E22",
            "Nguy cơ cao": "#C0392B",
        }
        risk_color = risk_colors.get(r.risk_level, "#7F8C8D")

        # Nhóm Denver dạng chuỗi
        group_str = "  ".join(
            f"<b>{g}</b>:{n}" for g, n in r.group_sizes.items() if n > 0
        )

        # Hội chứng
        if r.syndrome_flags:
            syndrome_html = "<ul style='margin:4px 0 0 16px;padding:0;'>" + "".join(
                f"<li style='color:#C0392B;'>{s}</li>" for s in r.syndrome_flags
            ) + "</ul>"
            syndrome_plain = "\n".join(f"  • {s}" for s in r.syndrome_flags)
        else:
            syndrome_html = "<span style='color:#27AE60;'>Không phát hiện hội chứng đặc trưng theo số lượng</span>"
            syndrome_plain = "Không phát hiện hội chứng đặc trưng theo số lượng"

        html_lines = [
            f"<b>① Số lượng NST:</b> <span style='font-size:20px;font-weight:900;color:{count_color};'>{total}</span>",
            f"<b>② Đánh giá:</b> <span style='color:{count_color};'>{count_label} (chuẩn {normal} ± {tolerance})</span>",
            f"<b>③ Giới tính ước tính:</b> {r.sex_estimation} "
            f"<span style='color:#7F8C8D;font-size:12px;'>(Độ tin cậy: {r.sex_confidence})</span>",
            f"<b>④ Mức độ rủi ro:</b> <span style='color:{risk_color};font-weight:bold;'>{r.risk_level}</span>",
            f"<b>⑤ Nhóm Denver:</b> <span style='font-size:13px;'>{group_str if group_str else '—'}</span>",
            f"<b>⑥ Hội chứng nghi ngờ:</b>{syndrome_html}",
            "<span style='color:#95A5A6;font-size:12px;'>⚠ Kết quả mang tính hỗ trợ — cần karyotype chuyên sâu để xác nhận.</span>",
        ]

        plain_lines = [
            f"1. Số lượng NST đếm được: {total}",
            f"2. Đánh giá: {count_label} (chuẩn {normal} ± {tolerance})",
            f"3. Giới tính ước tính: {r.sex_estimation} (Độ tin cậy: {r.sex_confidence})",
            f"4. Mức độ rủi ro: {r.risk_level}",
            f"5. Nhóm Denver: " + ", ".join(f"{g}:{n}" for g, n in r.group_sizes.items() if n > 0),
            f"6. Hội chứng nghi ngờ:\n{syndrome_plain}",
            "Lưu ý: Kết quả hỗ trợ chẩn đoán, không thay thế xét nghiệm chuyên sâu.",
        ]

        return "<br>".join(html_lines), "\n".join(plain_lines)