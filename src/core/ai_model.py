import os
import re
from ultralytics import YOLO


class ChromosomeAnalyzer:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.model_overlap = None
        self.model_anomaly = None
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
            print(f"⚠️ Định dạng {ext} chưa được tích hợp với Ultralytics YOLO: {path}")
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

    def analyze(self, image_path):
        if self.model is None:
            raise Exception("Model chưa được load! Kiểm tra đường dẫn trong config.yaml hoặc Cài đặt.")

        normal = int(self.config.get("normal_count", 46))
        tolerance = int(self.config.get("tolerance", 1))

        results = self.model.predict(source=image_path, conf=0.25, save=False)
        result = results[0]

        annotated_img = result.plot()
        total_count = len(result.boxes)

        # Có thể mở rộng: dùng model_overlap / model_anomaly khi pipeline được định nghĩa rõ
        _ = self.model_overlap
        _ = self.model_anomaly

        diff = abs(total_count - normal)
        is_normal = diff <= tolerance

        report = []
        report_plain = []

        report.append(f"<b>1. Tổng số nhiễm sắc thể đếm được:</b> <span style='font-size:18px;'>{total_count}</span>")
        report_plain.append(f"1. Tổng số NST đếm được: {total_count}")

        if is_normal:
            report.append(
                f"<b>2. Đánh giá số lượng:</b> <span style='color:#27AE60;'>Bình thường (chuẩn {normal} ± {tolerance})</span>"
            )
            report_plain.append(f"2. Đánh giá số lượng: Bình thường (chuẩn {normal} ± {tolerance})")
            report.append(
                "<b>3. Khuyến nghị:</b> Không phát hiện bất thường về số lượng trong ngưỡng đã cấu hình. "
                "Nên đối chiếu karyotype chi tiết khi cần đánh giá cấu trúc NST."
            )
            report_plain.append(
                "3. Khuyến nghị: Không phát hiện bất thường số lượng trong ngưỡng đã cấu hình."
            )
        else:
            status = "thừa" if total_count > normal else "thiếu"
            report.append(
                f"<b>2. Đánh giá số lượng:</b> <span style='color:#C0392B;'>Bất thường ({status}, lệch {diff} so với mức {normal} ± {tolerance})</span>"
            )
            report_plain.append(
                f"2. Đánh giá số lượng: Bất thường ({status}, lệch {diff} so với mức {normal} ± {tolerance})"
            )
            report.append(
                "<b>3. Khuyến nghị:</b> Có dấu hiệu lệch bội số lượng so với ngưỡng đã đặt. "
                "Cần xét nghiệm karyotype hoặc kỹ thuật di truyền phù hợp để xác định chi tiết."
            )
            report_plain.append(
                "3. Khuyến nghị: Có dấu hiệu lệch bội số lượng so với ngưỡng đã đặt — cần đánh giá thêm."
            )

        final_status_html = "<br>".join(report)
        final_status_plain = "\n".join(report_plain)
        return annotated_img, total_count, final_status_html, final_status_plain
