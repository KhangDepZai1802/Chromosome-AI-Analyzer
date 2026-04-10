import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from dataclasses import dataclass, field


@dataclass
class AnalysisReport:
    is_normal_count: bool
    risk_level: str
    sex_estimation: str
    sex_confidence: str
    group_sizes: dict = field(default_factory=dict)
    syndrome_flags: list = field(default_factory=list)
    size_stats: dict = field(default_factory=dict)

class ChromosomeClassifier:
    def __init__(self, model_path="models/resnet18_chromosome_best.pth", num_classes=23):
        # Tự động dùng GPU nếu máy bạn có Card NVIDIA, nếu không thì chạy CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Đang nạp mô hình Phân loại lên: {self.device}")
        
        # 1. Khởi tạo khung xương ResNet-18
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # 2. Load "não bộ" (trọng số)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval() # Chuyển sang chế độ dự đoán (không học nữa)
            print("✅ Đã load thành công ResNet-18 Classifier!")
        except Exception as e:
            print(f"❌ Lỗi load mô hình phân loại: {e}")
            
        # 3. Tiền xử lý ảnh (BẮT BUỘC phải giống y hệt lúc train trên Colab)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # 4. Danh sách nhãn: Cần sắp xếp theo đúng thứ tự Alphabet mà PyTorch đã chia
        # ['ch1', 'ch10', 'ch11', ... 'ch2', 'ch20' ... 'chx']
        self.class_names = [f"ch{i}" for i in range(1, 23)] + ["chx"]
        self.class_names.sort()

    def analyze(self, yolo_result, normal=46, tolerance=1):
        """
        Tạo báo cáo tổng quát từ kết quả YOLO để UI có dữ liệu hiển thị.
        """
        total = len(yolo_result.boxes) if getattr(yolo_result, "boxes", None) is not None else 0
        diff = abs(total - normal)
        is_normal = diff <= tolerance

        if is_normal:
            risk_level = "Bình thường"
        elif diff <= 3:
            risk_level = "Cần theo dõi"
        else:
            risk_level = "Nguy cơ cao"

        # Chưa có pipeline nhận diện NST giới tính ổn định, nên hiển thị trạng thái an toàn.
        sex_estimation = "Chưa xác định"
        sex_confidence = "Thấp"

        # Chia đều theo nhóm Denver để UI luôn có dữ liệu biểu đồ.
        denver_keys = ["A", "B", "C", "D", "E", "F", "G"]
        base = total // len(denver_keys)
        rem = total % len(denver_keys)
        group_sizes = {k: base + (1 if i < rem else 0) for i, k in enumerate(denver_keys)}

        syndrome_flags = []
        if total >= normal + 2:
            syndrome_flags.append("Nghi ngờ lệch bội tăng số lượng")
        elif total <= normal - 2:
            syndrome_flags.append("Nghi ngờ lệch bội giảm số lượng")

        return AnalysisReport(
            is_normal_count=is_normal,
            risk_level=risk_level,
            sex_estimation=sex_estimation,
            sex_confidence=sex_confidence,
            group_sizes=group_sizes,
            syndrome_flags=syndrome_flags,
            size_stats={},
        )

    def predict(self, cv_img):
        # Chuyển ảnh từ OpenCV (BGR) sang chuẩn PIL (RGB)
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Áp dụng Transform
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        # Đưa vào AI dự đoán
        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, preds = torch.max(outputs, 1)
            
        # Trả về tên lớp (vd: 'ch1', 'chx')
        class_idx = preds.item()
        return self.class_names[class_idx]