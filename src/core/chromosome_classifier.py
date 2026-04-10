# src/core/chromosome_classifier.py
"""
Module phân tích chuyên sâu NST dựa trên kết quả YOLO instance segmentation.
Phân tích hình học mask để ước tính:
  - Kích thước tương đối từng NST
  - Ước tính giới tính (XX/XY) dựa vào đặc trưng hình học
  - Phân loại hội chứng di truyền theo số lượng
  - Nhóm NST theo kích thước (A-G theo Denver)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class ChromosomeFeature:
    """Đặc trưng hình học của 1 NST"""
    index: int
    area: float          # Diện tích mask (px²)
    length: float        # Chiều dài trục chính (px)
    width: float         # Chiều rộng trục phụ (px)
    aspect_ratio: float  # length / width
    centromere_idx: float  # Vị trí tâm động tương đối (0-1), ước tính từ hình dạng
    contour: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class AnalysisReport:
    total_count: int
    normal_count: int       # Ngưỡng chuẩn
    tolerance: int
    is_normal_count: bool

    # Phân tích giới tính
    sex_estimation: str        # "XX (Nữ)", "XY (Nam)", "Không xác định"
    sex_confidence: str        # "Cao", "Trung bình", "Thấp"

    # Hội chứng
    syndrome_flags: List[str]  # Danh sách hội chứng nghi ngờ
    risk_level: str            # "Bình thường", "Cần theo dõi", "Nguy cơ cao"

    # Nhóm Denver
    group_sizes: dict          # {"A": 6, "B": 4, ...}

    # Thống kê hình học
    size_stats: dict           # mean_area, std_area, size_distribution

    features: List[ChromosomeFeature] = field(default_factory=list)


class ChromosomeClassifier:
    """
    Phân tích NST dựa thuần túy vào hình học mask từ YOLO segment.
    Không cần model phân loại riêng.
    """

    # Ngưỡng Denver (tỷ lệ tương đối, chuẩn hoá theo NST lớn nhất = 1.0)
    # Chia 7 nhóm A-G theo chiều dài tương đối
    DENVER_GROUPS = {
        "A": (0.75, 1.01),   # NST 1-3: lớn nhất
        "B": (0.60, 0.75),   # NST 4-5
        "C": (0.45, 0.60),   # NST 6-12 + X
        "D": (0.35, 0.45),   # NST 13-15 (tâm đầu)
        "E": (0.25, 0.35),   # NST 16-18
        "F": (0.18, 0.25),   # NST 19-20
        "G": (0.00, 0.18),   # NST 21-22 + Y: nhỏ nhất
    }

    # Bảng hội chứng theo số lượng NST
    SYNDROME_TABLE = {
        45: [
            ("Turner (45,X)", "Thiếu 1 NST giới tính, thường gặp ở nữ", "Nguy cơ cao")
        ],
        47: [
            ("Down (47,+21)", "Thừa NST số 21 — cần xác nhận bằng karyotype", "Nguy cơ cao"),
            ("Klinefelter (47,XXY)", "Thừa NST X ở nam — nếu phát hiện XXY", "Nguy cơ cao"),
            ("47,XYY", "Thừa NST Y ở nam", "Cần theo dõi"),
            ("47,XXX", "Thừa NST X ở nữ", "Cần theo dõi"),
        ],
        48: [
            ("48,XXXY / 48,XXYY", "Đa thể NST giới tính hiếm gặp", "Nguy cơ cao"),
        ],
        49: [
            ("49,XXXXY / 49,XXXXX", "Lệch bội nặng, rất hiếm", "Nguy cơ cao"),
        ],
    }

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def analyze(
        self,
        yolo_result,
        normal_count: int = 46,
        tolerance: int = 1,
    ) -> AnalysisReport:
        """
        Nhận kết quả YOLO (results[0]) và trả về AnalysisReport chi tiết.
        """
        features = self._extract_features(yolo_result)
        total = len(features)
        is_normal = abs(total - normal_count) <= tolerance

        group_sizes = self._classify_denver_groups(features)
        sex_est, sex_conf = self._estimate_sex(features, total)
        syndromes = self._detect_syndromes(total, sex_est)
        risk = self._compute_risk(is_normal, syndromes)
        size_stats = self._compute_size_stats(features)

        return AnalysisReport(
            total_count=total,
            normal_count=normal_count,
            tolerance=tolerance,
            is_normal_count=is_normal,
            sex_estimation=sex_est,
            sex_confidence=sex_conf,
            syndrome_flags=syndromes,
            risk_level=risk,
            group_sizes=group_sizes,
            size_stats=size_stats,
            features=features,
        )

    # ------------------------------------------------------------------
    # PRIVATE: Trích xuất đặc trưng hình học
    # ------------------------------------------------------------------

    def _extract_features(self, yolo_result) -> List[ChromosomeFeature]:
        features = []
        masks = None

        # Lấy masks từ YOLO segment result
        if yolo_result.masks is not None:
            try:
                masks_data = yolo_result.masks.data.cpu().numpy()  # (N, H, W)
                masks = masks_data
            except Exception:
                masks = None

        boxes = yolo_result.boxes
        n = len(boxes) if boxes is not None else 0

        for i in range(n):
            feat = self._analyze_single(i, masks, yolo_result)
            features.append(feat)

        # Chuẩn hoá area theo NST lớn nhất (nếu có)
        if features:
            max_area = max(f.area for f in features) or 1.0
            for f in features:
                f.area = f.area / max_area  # Tỷ lệ 0-1

        return features

    def _analyze_single(self, idx: int, masks, yolo_result) -> ChromosomeFeature:
        """Phân tích 1 NST từ mask hoặc bounding box."""
        area = 0.0
        length = 1.0
        width = 1.0
        centromere_idx = 0.5
        contour = None

        try:
            if masks is not None and idx < len(masks):
                mask_bin = (masks[idx] > 0.5).astype(np.uint8)
                area = float(np.sum(mask_bin))

                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    contour = cnt
                    if len(cnt) >= 5:
                        ellipse = cv2.fitEllipse(cnt)
                        length = max(ellipse[1])
                        width = min(ellipse[1]) + 1e-6
                    else:
                        rect = cv2.boundingRect(cnt)
                        length = max(rect[2], rect[3])
                        width = min(rect[2], rect[3]) + 1e-6
                    # Ước tính centromere index bằng skeleton
                    centromere_idx = self._estimate_centromere(mask_bin)
            else:
                # Fallback: dùng bounding box
                box = yolo_result.boxes[idx]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width_bb = x2 - x1
                height_bb = y2 - y1
                area = width_bb * height_bb
                length = max(width_bb, height_bb)
                width = min(width_bb, height_bb) + 1e-6
        except Exception as e:
            pass

        aspect_ratio = length / width if width > 0 else 1.0

        return ChromosomeFeature(
            index=idx,
            area=area,
            length=length,
            width=width,
            aspect_ratio=aspect_ratio,
            centromere_idx=centromere_idx,
            contour=contour,
        )

    def _estimate_centromere(self, mask_bin: np.ndarray) -> float:
        """
        Ước tính vị trí tâm động bằng cách phân tích độ co thắt của mask.
        Trả về giá trị 0-1 (0 = tâm đầu, 0.5 = tâm giữa).
        """
        try:
            # Tìm hướng trục chính bằng PCA
            pts = np.argwhere(mask_bin > 0).astype(np.float32)
            if len(pts) < 10:
                return 0.5

            mean, eigvec = cv2.PCACompute(pts, mean=None)
            # Project lên trục chính
            proj = (pts - mean) @ eigvec.T
            proj_1d = proj[:, 0]

            # Chia thành 10 đoạn dọc theo trục chính, đo độ rộng mỗi đoạn
            bins = np.linspace(proj_1d.min(), proj_1d.max(), 11)
            widths = []
            for j in range(10):
                mask_seg = (proj_1d >= bins[j]) & (proj_1d < bins[j + 1])
                perp = proj[mask_seg, 1]
                w = perp.max() - perp.min() if len(perp) > 1 else 0
                widths.append(w)

            if not widths or max(widths) == 0:
                return 0.5

            # Vị trí co thắt nhất = tâm động ước tính
            min_idx = int(np.argmin(widths))
            return (min_idx + 0.5) / 10.0
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # PRIVATE: Phân nhóm Denver
    # ------------------------------------------------------------------

    def _classify_denver_groups(self, features: List[ChromosomeFeature]) -> dict:
        """Phân nhóm NST theo kích thước tương đối (Denver A-G)."""
        counts = {g: 0 for g in self.DENVER_GROUPS}
        if not features:
            return counts

        # area đã chuẩn hoá 0-1
        for f in features:
            for group, (lo, hi) in self.DENVER_GROUPS.items():
                if lo <= f.area < hi:
                    counts[group] += 1
                    break
        return counts

    # ------------------------------------------------------------------
    # PRIVATE: Ước tính giới tính
    # ------------------------------------------------------------------

    def _estimate_sex(
        self, features: List[ChromosomeFeature], total: int
    ) -> Tuple[str, str]:
        """
        Ước tính giới tính dựa trên phân phối kích thước NST.

        Nguyên lý:
        - NST X ~ kích thước trung bình (nhóm C)
        - NST Y ~ kích thước rất nhỏ (nhóm G) + tỷ lệ aspect ratio cao
        - Nếu phát hiện ≥1 NST cực nhỏ + dài → nghi Y → XY (Nam)
        - Nếu không → XX (Nữ)

        Đây là ước tính heuristic, độ chính xác phụ thuộc chất lượng ảnh.
        """
        if not features or total < 40:
            return "Không xác định (thiếu NST)", "Thấp"

        areas = [f.area for f in features]
        mean_a = np.mean(areas)
        std_a = np.std(areas)

        # NST rất nhỏ (dưới mean - 1.5*std) và có aspect ratio > 2.5 → ứng viên Y
        y_candidates = [
            f for f in features
            if f.area < (mean_a - 1.5 * std_a) and f.aspect_ratio > 2.0
        ]

        # NST nhỏ nhưng không quá nhỏ (nhóm G chung: Y hoặc 21/22)
        small_nsT = [f for f in features if f.area < mean_a - std_a]

        if len(y_candidates) >= 1:
            conf = "Trung bình" if len(y_candidates) == 1 else "Thấp"
            return "XY — Nam (ước tính)", conf
        elif len(small_nsT) <= 2:
            # Ít NST nhỏ → có thể không có Y
            return "XX — Nữ (ước tính)", "Trung bình"
        else:
            return "Không xác định rõ", "Thấp"

    # ------------------------------------------------------------------
    # PRIVATE: Phát hiện hội chứng
    # ------------------------------------------------------------------

    def _detect_syndromes(self, total: int, sex_est: str) -> List[str]:
        flags = []
        if total in self.SYNDROME_TABLE:
            for name, desc, _ in self.SYNDROME_TABLE[total]:
                # Lọc thêm dựa vào giới tính ước tính
                if "Nam" in name and "Nữ" in sex_est:
                    continue
                if "Nữ" in name and "Nam" in sex_est:
                    continue
                flags.append(f"{name}: {desc}")

        # Trường hợp số lệch nhiều
        if total < 44:
            flags.append(f"Thiếu hụt nghiêm trọng ({total} NST): có thể mẫu không đạt hoặc lệch bội nặng")
        elif total > 50:
            flags.append(f"Quá nhiều NST ({total}): nghi ngờ đa bội hoặc lỗi phân đoạn")

        return flags

    def _compute_risk(self, is_normal: bool, syndromes: List[str]) -> str:
        if not is_normal and syndromes:
            return "Nguy cơ cao"
        elif not is_normal:
            return "Cần theo dõi"
        elif syndromes:
            return "Cần theo dõi"
        return "Bình thường"

    # ------------------------------------------------------------------
    # PRIVATE: Thống kê
    # ------------------------------------------------------------------

    def _compute_size_stats(self, features: List[ChromosomeFeature]) -> dict:
        if not features:
            return {}
        areas = [f.area for f in features]
        return {
            "mean_area": float(np.mean(areas)),
            "std_area": float(np.std(areas)),
            "min_area": float(np.min(areas)),
            "max_area": float(np.max(areas)),
            "cv_percent": float(np.std(areas) / np.mean(areas) * 100) if np.mean(areas) > 0 else 0,
        }
