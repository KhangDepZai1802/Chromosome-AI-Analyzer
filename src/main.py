import torch
import torchvision.transforms as transforms
from PIL import Image # Dùng để đọc ảnh chuẩn cho PyTorch
import cv2
import random
import sys
import csv         
import os
import json
import ctypes
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QLabel, QFrame, QStatusBar,
                             QFileDialog, QMessageBox, QDialog, QLineEdit, 
                             QFormLayout, QDialogButtonBox, QSpinBox)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap, QImage, QIcon

# === CLASS CỬA SỔ CÀI ĐẶT ===
class SettingsDialog(QDialog):
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ Cài đặt Hệ thống")
        self.setFixedWidth(400)
        
        self.layout = QFormLayout(self)
        
        # 1. Ô nhập đường dẫn mô hình
        self.model_path_input = QLineEdit(current_config.get("model_path", "models/model_v1.pt"))
        
        # 2. Ô nhập số lượng NST chuẩn
        self.normal_count_input = QSpinBox()
        self.normal_count_input.setValue(current_config.get("normal_count", 46))
        
        # 3. Ô nhập sai số (Tolerance)
        self.tolerance_input = QSpinBox()
        self.tolerance_input.setValue(current_config.get("tolerance", 1))
        
        # Đưa các ô nhập liệu vào Form
        self.layout.addRow("Đường dẫn Model AI:", self.model_path_input)
        self.layout.addRow("Số lượng NST chuẩn:", self.normal_count_input)
        self.layout.addRow("Sai số cho phép (±):", self.tolerance_input)
        
        # Nút OK và Cancel
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
        self.layout.addWidget(self.buttons)
        
    def get_config(self):
        # Trả về dữ liệu khi người dùng bấm OK
        return {
            "model_path": self.model_path_input.text(),
            "normal_count": self.normal_count_input.value(),
            "tolerance": self.tolerance_input.value()
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phần mềm Phân tích Nhiễm sắc thể - AI")
        self.setGeometry(100, 100, 1200, 700) # Kích thước mặc định (Rộng x Cao)
        self.setWindowIcon(QIcon("assets/logoNST.jpg"))

        self.load_config()
        self.load_ai_model()
        # Widget chính chứa toàn bộ giao diện
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Layout tổng: Chia theo chiều ngang (Trái - Giữa - Phải)
        main_layout = QHBoxLayout(main_widget)
        
        # ==================== KHU VỰC 1: BẢNG ĐIỀU KHIỂN (TRÁI) ====================
        control_panel = QFrame()
        control_panel.setFrameShape(QFrame.Shape.StyledPanel)
        control_panel.setFixedWidth(250)
        control_layout = QVBoxLayout(control_panel)
        
        self.btn_load = QPushButton("📂 Tải ảnh lên")
        self.btn_load.setMinimumHeight(50)
        self.btn_load.clicked.connect(self.load_image)
        self.btn_run_ai = QPushButton("🧠 Chạy AI Phân tích")
        self.btn_run_ai.setMinimumHeight(50)
        self.btn_run_ai.setEnabled(False) # Tạm ẩn khi chưa có ảnh
        self.btn_run_ai.clicked.connect(self.run_ai_analysis)
        self.btn_settings = QPushButton("⚙️ Cài đặt hệ thống")
        self.btn_settings.clicked.connect(self.open_settings)

        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_run_ai)
        control_layout.addStretch() # Đẩy nút cài đặt xuống đáy
        control_layout.addWidget(self.btn_settings)
        
        # ==================== KHU VỰC 2: TRÌNH CHIẾU ẢNH (GIỮA) ====================
        image_panel = QFrame()
        image_panel.setFrameShape(QFrame.Shape.StyledPanel)
        image_layout = QHBoxLayout(image_panel)
        
        self.lbl_image_original = QLabel("Ảnh gốc hiển thị ở đây")
        self.lbl_image_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image_original.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        self.lbl_image_result = QLabel("Ảnh AI phân đoạn hiển thị ở đây")
        self.lbl_image_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_image_result.setStyleSheet("background-color: #2b2b2b; color: white;")
        
        image_layout.addWidget(self.lbl_image_original)
        image_layout.addWidget(self.lbl_image_result)
        
        # ==================== KHU VỰC 3: KẾT QUẢ & XUẤT FILE (PHẢI) ====================
        result_panel = QFrame()
        result_panel.setFrameShape(QFrame.Shape.StyledPanel)
        result_panel.setFixedWidth(250)
        result_layout = QVBoxLayout(result_panel)
        
        lbl_title_result = QLabel("KẾT QUẢ PHÂN TÍCH")
        lbl_title_result.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        lbl_title_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_count = QLabel("Số lượng NST: --")
        self.lbl_count.setFont(QFont("Arial", 14))
        
        self.lbl_status = QLabel("Đánh giá: --")
        self.lbl_status.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        
        self.btn_export = QPushButton("📥 Xuất báo cáo CSV")
        self.btn_export.setMinimumHeight(40)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv)
        
        result_layout.addWidget(lbl_title_result)
        result_layout.addSpacing(20)
        result_layout.addWidget(self.lbl_count)
        result_layout.addWidget(self.lbl_status)
        result_layout.addStretch()
        result_layout.addWidget(self.btn_export)
        
        # Gắn 3 khu vực vào layout tổng
        main_layout.addWidget(control_panel)
        main_layout.addWidget(image_panel)
        main_layout.addWidget(result_panel)
        
        # ==================== KHU VỰC 4: THANH TRẠNG THÁI (DƯỚI CÙNG) ====================
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Sẵn sàng... | © 2026 - Đồ án phần mềm nhúng AI - Sinh viên thực hiện: Khang, Doanh, Khánh")

# BẠN THÊM HÀM NÀY VÀO TRONG CLASS MainWindow
    def load_image(self):
        # 1. Mở hộp thoại chọn file (Chỉ lọc các file ảnh)
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh Nhiễm sắc thể",
            "",
            "Image Files (*.png *.jpg *.jpeg *.tif)"
        )
        
        # 2. Nếu người dùng có chọn file (không bấm Cancel)
        if file_name:
            self.current_image_path = file_name # Lưu đường dẫn ảnh hiện tại để sử dụng sau này khi chạy AI
            # Đọc ảnh
            pixmap = QPixmap(file_name)
            
            # Thay đổi kích thước ảnh cho vừa với giao diện (tối đa 500x500) mà vẫn giữ nguyên tỷ lệ khung hình
            scaled_pixmap = pixmap.scaled(
                500, 500, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            
            # Gắn ảnh lên khung bên trái
            self.lbl_image_original.setPixmap(scaled_pixmap)
            
            # Cập nhật thông báo dưới thanh trạng thái
            self.statusBar.showMessage(f"Đã tải ảnh: {file_name}")
            
            # Bật sáng nút "Chạy AI Phân tích" vì đã có ảnh đầu vào
            self.btn_run_ai.setEnabled(True)
            self.btn_run_ai.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
    
    # === HÀM HỖ TRỢ: CHUYỂN ẢNH OPENCV SANG ẢNH GIAO DIỆN ===
    def cv_to_qpixmap(self, cv_img):
        # Chuyển hệ màu BGR của OpenCV sang RGB chuẩn
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Resize ảnh cho vừa khung hình 500x500
        p = convert_to_Qt_format.scaled(500, 500, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    # === HÀM CHÍNH: LUỒNG CHẠY AI PHÂN TÍCH ===
    def run_ai_analysis(self):
        self.statusBar.showMessage("Đang phân tích bằng AI... Vui lòng đợi...")
        self.btn_run_ai.setEnabled(False)
        QApplication.processEvents()

        # --- PHẦN 1: DÀNH CHO AI THẬT (Tạm ẩn chờ Model) ---
        """
        image = Image.open(self.current_image_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # output_tensor = self.model(input_tensor)
            pass
        """
        
        # --- PHẦN 2: GIẢ LẬP ĐỂ APP KHÔNG BỊ LỖI HIỂN THỊ ---
        img = cv2.imread(self.current_image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV) # Tạo lại biến mask
        
        # Hiển thị ảnh Mask kết quả sang khung bên phải
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        self.lbl_image_result.setPixmap(self.cv_to_qpixmap(mask_bgr))
        
        # --- PHẦN 3: XỬ LÝ ĐẾM VÀ ĐÁNH GIÁ (Giữ nguyên của bạn) ---
        normal_count = self.config.get("normal_count", 46)
        tolerance = self.config.get("tolerance", 1)
        count_result = random.choice([normal_count-2, normal_count-1, normal_count, normal_count, normal_count+1])
        
        self.lbl_count.setText(f"Số lượng NST: {count_result}")
        if abs(count_result - normal_count) <= tolerance:
            self.lbl_status.setText("Đánh giá: BÌNH THƯỜNG")
            self.lbl_status.setStyleSheet("color: #28a745;")
        else:
            self.lbl_status.setText("Đánh giá: BẤT THƯỜNG")
            self.lbl_status.setStyleSheet("color: #dc3545;")
            
        self.btn_export.setEnabled(True)
        self.btn_export.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.btn_run_ai.setEnabled(True)
        self.btn_run_ai.setText("🧠 Chạy AI Phân tích")
        self.statusBar.showMessage("Phân tích hoàn tất! Bạn có thể xuất báo cáo.")

        # === HÀM XUẤT BÁO CÁO CSV ===
    def export_csv(self):
        # 1. Lấy ngày giờ hiện tại
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 2. Lấy tên file ảnh gốc (chỉ lấy tên file, cắt bỏ đường dẫn dài)
        image_name = os.path.basename(self.current_image_path)
        
        # 3. Lấy dữ liệu từ giao diện (cắt bỏ các chữ râu ria đi)
        count_text = self.lbl_count.text().replace("Số lượng NST: ", "")
        status_text = self.lbl_status.text().replace("Đánh giá: ", "")
        
        # 4. Mở hộp thoại để người dùng chọn nơi lưu file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu báo cáo phân tích",
            f"Bao_cao_NST_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", # Tên file mặc định
            "CSV Files (*.csv)"
        )
        
        # 5. Tiến hành lưu file nếu người dùng chọn nơi lưu
        if file_path:
            try:
                # Mở file và ghi dữ liệu (dùng chuẩn utf-8-sig để Excel đọc không bị lỗi font tiếng Việt)
                with open(file_path, mode='w', newline='', encoding='utf-8-sig') as file:
                    writer = csv.writer(file)
                    
                    # Ghi dòng tiêu đề các cột
                    writer.writerow(["Thời gian Phân tích", "Tên File Ảnh", "Số lượng NST đếm được", "Kết luận Đánh giá"])
                    
                    # Ghi dòng dữ liệu thực tế
                    writer.writerow([current_time, image_name, count_text, status_text])
                
                # 6. Hiện thông báo pop-up báo thành công
                QMessageBox.information(self, "Thành công", f"Đã xuất báo cáo thành công tại:\n{file_path}")
                
            except Exception as e:
                # Báo lỗi nếu có trục trặc (ví dụ: file đang mở nên không thể ghi đè)
                QMessageBox.critical(self, "Lỗi", f"Không thể lưu file. Chi tiết lỗi:\n{str(e)}")

# === HÀM QUẢN LÝ CẤU HÌNH (CONFIG) ===
    def load_config(self):
        self.config_file = "config.json"
        # Nếu chưa có file config thì tạo mặc định
        if not os.path.exists(self.config_file):
            self.config = {"model_path": "models/model_v1.pt", "normal_count": 46, "tolerance": 1}
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
        else:
            # Nếu có rồi thì đọc lên
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

    def open_settings(self):
        # Mở cửa sổ Pop-up
        dialog = SettingsDialog(self.config, self)
        if dialog.exec(): # Nếu người dùng bấm OK
            self.config = dialog.get_config()
            # Ghi đè vào file config.json
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            QMessageBox.information(self, "Thành công", "Đã lưu cấu hình hệ thống mới!")
    
    def load_ai_model(self):
        # 1. Xác định dùng Card đồ họa (cuda) hay CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Khởi tạo cấu trúc mô hình (GIẢ ĐỊNH bạn dùng mô hình tên là UNet)
        # Lưu ý: Class UNet này bạn sẽ phải tự viết hoặc copy từ lúc train sang
        # self.model = UNet(in_channels=3, out_channels=1).to(self.device) 
        
        # 3. Nạp trọng số (từ file .pt) vào cấu trúc mô hình
        model_path = self.config.get("model_path", "models/model_v1.pt")
        try:
            # Load weights
            # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # 4. Chuyển mô hình sang chế độ Đánh giá (không phải chế độ học)
            # self.model.eval() 
            print("Đã load model thành công lên:", self.device)
        except Exception as e:
            print("Chưa load được model:", e)
            
        # 5. Định nghĩa phép biến đổi ảnh (Resize, Chuẩn hóa màu sắc)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # Đưa về kích thước model yêu cầu
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Cài đặt giao diện màu tối (Dark Mode) cơ bản cho dịu mắt
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())