# src/ui/main_window.py
import os
import yaml
import cv2
from datetime import datetime
from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, 
                             QVBoxLayout, QPushButton, QLabel, QFrame, QStatusBar,
                             QFileDialog, QMessageBox, QDialog, QLineEdit, 
                             QFormLayout, QDialogButtonBox, QSpinBox, QApplication,
                             QGraphicsDropShadowEffect, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QIcon, QColor, QAction

from src.core.ai_model import ChromosomeAnalyzer
from src.core.exporter import Exporter

class SettingsDialog(QDialog):
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt Hệ thống")
        self.setFixedWidth(520)
        
        # 1. CSS (QSS) thiết kế riêng cho Popup
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFFFF;
            }
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #34495E;
            }
            /* Giao diện ô nhập liệu */
            QLineEdit, QSpinBox {
                padding: 10px;
                border: 2px solid #ECF0F1;
                border-radius: 8px;
                background-color: #F8F9FA;
                font-size: 14px;
                color: #2C3E50;
            }
            /* Hiệu ứng khi bấm chuột vào ô nhập liệu */
            QLineEdit:focus, QSpinBox:focus {
                border: 2px solid #3498DB; /* Đổi viền sang màu xanh dương */
                background-color: #FFFFFF;
            }
            /* Reset style của SpinBox (nút tăng giảm) cho đẹp hơn */
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
                border-radius: 4px;
            }
            /* Nút bấm tổng quát */
            QPushButton {
                min-height: 40px;
                min-width: 120px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
                border: none;
            }
        """)
        
        # 2. Layout chính (Dọc)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30) # Căn lề rộng rãi
        main_layout.setSpacing(25)
        
        # Tiêu đề bên trong Popup
        header_label = QLabel("CẤU HÌNH HỆ THỐNG")
        header_label.setStyleSheet("font-size: 18px; font-weight: 900; color: #2C3E50; border-bottom: 2px solid #ECF0F1; padding-bottom: 10px;")
        main_layout.addWidget(header_label)
        
        # 3. Form Layout (Chứa các dòng nhập liệu)
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        
        self.model_path_input = QLineEdit(current_config.get("model_path", "models/best.pt"))
        self.model_overlap_input = QLineEdit(current_config.get("model_overlap_path", "") or "")
        self.model_anomaly_input = QLineEdit(current_config.get("model_anomaly_path", "") or "")
        self.normal_count_input = QSpinBox()
        self.normal_count_input.setRange(0, 200)
        self.normal_count_input.setValue(current_config.get("normal_count", 46))
        self.tolerance_input = QSpinBox()
        self.tolerance_input.setRange(0, 50)
        self.tolerance_input.setValue(current_config.get("tolerance", 1))
        
        form_layout.addRow("Mô hình phân đoạn / đếm NST:", self.model_path_input)
        form_layout.addRow("Mô hình tách NST chồng (tuỳ chọn):", self.model_overlap_input)
        form_layout.addRow("Mô hình hỗ trợ bất thường (tuỳ chọn):", self.model_anomaly_input)
        form_layout.addRow("Số NST chuẩn (2n):", self.normal_count_input)
        form_layout.addRow("Sai số cho phép (±):", self.tolerance_input)
        
        main_layout.addLayout(form_layout)
        
        # 4. Thanh chứa nút bấm (Ngang)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch() # Đẩy các nút sang góc phải
        
        # Tạo nút Hủy
        btn_cancel = QPushButton("Hủy bỏ")
        btn_cancel.setStyleSheet("""
            QPushButton { background-color: #ECF0F1; color: #7F8C8D; }
            QPushButton:hover { background-color: #BDC3C7; color: white; }
        """)
        btn_cancel.clicked.connect(self.reject)
        
        # Tạo nút Lưu
        btn_save = QPushButton("💾 Lưu cấu hình")
        btn_save.setStyleSheet("""
            QPushButton { background-color: #00B894; color: white; }
            QPushButton:hover { background-color: #009688; }
        """)
        btn_save.clicked.connect(self.accept)
        
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_save)
        
        main_layout.addLayout(btn_layout)
        
    def get_config(self):
        return {
            "model_path": self.model_path_input.text().strip(),
            "model_overlap_path": self.model_overlap_input.text().strip(),
            "model_anomaly_path": self.model_anomaly_input.text().strip(),
            "normal_count": self.normal_count_input.value(),
            "tolerance": self.tolerance_input.value(),
        }

class WorkerThread(QThread):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, analyzer, path):
        super().__init__()
        self.analyzer = analyzer
        self.path = path

    def run(self):
        try:
            result = self.analyzer.analyze(self.path)
            self.finished.emit(result)
        except Exception as e:
            self.failed.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phần mềm Phân tích Nhiễm sắc thể - AI")
        self.setGeometry(100, 100, 1300, 800)
        self.setWindowIcon(QIcon("assets/logoNST.png"))

        self.current_image_path = None
        self.last_result_bgr = None
        self.last_count = None
        self.last_report_plain = ""
        self.load_config()
        self.ai_analyzer = ChromosomeAnalyzer(self.config)
        
        self.init_ui()
        self._build_menu_bar()
        self.apply_stylesheet() # Gọi hàm áp dụng giao diện Light Medical

        # Khởi động đồng hồ thời gian thực
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000) # Cập nhật mỗi 1 giây
        self.update_clock()

    def init_ui(self):
        # Widget chính
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        # Layout tổng dọc (Header phía trên, Nội dung phía dưới)
        master_layout = QVBoxLayout(main_widget)
        master_layout.setContentsMargins(20, 20, 20, 20)
        master_layout.setSpacing(20)

        # ==================== HEADER (Tiêu đề & Logo) ====================
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # 1. Logo
        self.lbl_logo = QLabel()
        if os.path.exists("assets/logoNST.png"):
            pixmap = QPixmap("assets/logoNST.png").scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.lbl_logo.setPixmap(pixmap)
        else:
            self.lbl_logo.setText("🏥")
            self.lbl_logo.setFont(QFont("Arial", 30))
            
        # 2. Tên phần mềm
        self.lbl_title = QLabel("HỆ THỐNG PHÂN TÍCH NHIỄM SẮC THỂ AI")
        self.lbl_title.setObjectName("HeaderTitle") # Đặt tên để style
        
        # 3. Đồng hồ thời gian thực
        self.lbl_clock = QLabel("00:00:00")
        self.lbl_clock.setObjectName("HeaderClock")
        self.lbl_clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        
        header_layout.addWidget(self.lbl_logo)
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.lbl_clock)
        
        master_layout.addWidget(header_frame)

        # ==================== BODY (Nội dung chính ngang) ====================
        body_layout = QHBoxLayout()
        body_layout.setSpacing(20)
        master_layout.addLayout(body_layout)

        # --- Khu vực 1: Bảng điều khiển ---
        control_panel = QFrame()
        control_panel.setObjectName("CardFrame") # Đặt tên để biến thành Card bo góc
        self.add_shadow(control_panel)
        control_panel.setFixedWidth(250)
        control_layout = QVBoxLayout(control_panel)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)
        
        lbl_control_title = QLabel("BẢNG ĐIỀU KHIỂN")
        lbl_control_title.setObjectName("CardTitle")
        
        self.btn_load = QPushButton("📂 Tải ảnh lên")
        self.btn_load.setObjectName("ActionBtn")
        self.btn_load.clicked.connect(self.load_image)
        
        self.btn_run_ai = QPushButton("🧠 Chạy AI Phân tích")
        self.btn_run_ai.setObjectName("AiBtn")
        self.btn_run_ai.setEnabled(False)
        self.btn_run_ai.clicked.connect(self.run_ai_analysis)
        
        # --- THÊM DÒNG NÀY: Nhãn trạng thái ngay dưới nút ---
        self.lbl_analysis_status = QLabel("Trạng thái: Sẵn sàng")
        self.lbl_analysis_status.setObjectName("AnalysisStatus")
        self.lbl_analysis_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_reset = QPushButton("🔄 Tải lại (Reset)")
        self.btn_reset.setObjectName("ResetBtn")
        self.btn_reset.setEnabled(False) # Mặc định bị khóa khi chưa có ảnh
        self.btn_reset.clicked.connect(self.reset_all)
        
        # Biến quản lý hiệu ứng dấu chấm
        self.dot_count = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation_text)

        self.btn_settings = QPushButton("⚙️ Cài đặt hệ thống")
        self.btn_settings.setObjectName("SettingBtn")
        self.btn_settings.clicked.connect(self.open_settings)

        control_layout.addWidget(lbl_control_title)
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_run_ai)

        # --- THÊM DÒNG NÀY VÀO ĐÂY ---
        control_layout.addWidget(self.lbl_analysis_status)
        control_layout.addWidget(self.btn_reset)

        control_layout.addStretch()
        control_layout.addWidget(self.btn_settings)
        
        # --- Khu vực 2: Trình chiếu ảnh ---
        image_panel = QFrame()
        image_panel.setObjectName("CardFrame")
        self.add_shadow(image_panel)
        image_layout = QHBoxLayout(image_panel)
        image_layout.setContentsMargins(15, 15, 15, 15)
        image_layout.setSpacing(15)
        
        self.lbl_image_original = QLabel("Ảnh gốc (Chưa tải)")
        self.lbl_image_result = QLabel("Kết quả AI phân tích")
        
        for lbl in [self.lbl_image_original, self.lbl_image_result]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setObjectName("ImageContainer")
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            image_layout.addWidget(lbl)
            
        # --- Khu vực 3: Kết quả ---
        result_panel = QFrame()
        result_panel.setObjectName("CardFrame")
        self.add_shadow(result_panel)
        result_panel.setFixedWidth(280)
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(20, 20, 20, 20)
        result_layout.setSpacing(15)
        
        lbl_title_result = QLabel("KẾT QUẢ PHÂN TÍCH")
        lbl_title_result.setObjectName("CardTitle")
        
        self.lbl_count = QLabel("Kết quả đếm:\n--") # Rút ngắn tiêu đề mặc định
        self.lbl_count.setObjectName("ResultLabel")
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_count.setWordWrap(True) # Kích hoạt xuống dòng ngay từ đầu
        
        # (Khu vực 3 của hàm init_ui)
        self.lbl_status = QLabel("Báo cáo Y khoa:\n--")
        self.lbl_status.setObjectName("ResultLabel")
        
        # --- THÊM 2 DÒNG NÀY ĐỂ BÁO CÁO KHÔNG BỊ TRÀN KHUNG ---
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # -----------------------------------------------------
        
        self.btn_export = QPushButton("📥 Xuất báo cáo CSV")
        self.btn_export.setObjectName("ExportBtn")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv)

        self.btn_save_bundle = QPushButton("💾 Lưu ảnh & báo cáo")
        self.btn_save_bundle.setObjectName("ExportBtn")
        self.btn_save_bundle.setEnabled(False)
        self.btn_save_bundle.setToolTip("Lưu ảnh gốc, ảnh phân đoạn và file báo cáo .txt vào một thư mục")
        self.btn_save_bundle.clicked.connect(self.save_result_bundle)
        
        result_layout.addWidget(lbl_title_result)
        result_layout.addSpacing(20)
        result_layout.addWidget(self.lbl_count)
        result_layout.addWidget(self.lbl_status)
        result_layout.addStretch()
        result_layout.addWidget(self.btn_save_bundle)
        result_layout.addWidget(self.btn_export)
        
        # Ráp 3 thẻ vào Body
        body_layout.addWidget(control_panel)
        body_layout.addWidget(image_panel, 1) # Cho thẻ ảnh chiếm nhiều không gian nhất
        body_layout.addWidget(result_panel)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(
            "Trình tự: Tải ảnh → Chạy AI phân tích → Xem đếm & đánh giá → Lưu / xuất kết quả"
        )
        self._copyright_label = QLabel("© Đồ án NST-AI — Phần mềm hỗ trợ nghiên cứu / giáo dục")
        self._copyright_label.setStyleSheet("color: #7F8C8D; font-size: 11px; padding-right: 8px;")
        self.statusBar.addPermanentWidget(self._copyright_label)

    def _build_menu_bar(self):
        menu = self.menuBar()
        help_menu = menu.addMenu("Trợ giúp")
        act_about = QAction("Giới thiệu & bản quyền", self)
        act_about.triggered.connect(self.show_about_copyright)
        help_menu.addAction(act_about)

    def show_about_copyright(self):
        self.show_modern_message(
            "Giới thiệu & bản quyền",
            "Phần mềm phân tích NST tích hợp mô hình AI — đồ án môn học.\n\n"
            "Ứng dụng hỗ trợ tải ảnh hiển thị, phân đoạn/đếm NST và đánh giá số lượng "
            "theo ngưỡng cấu hình (ví dụ 46 ± 1).\n\n"
            "© Bản quyền thuộc tác giả dự án. Không sử dụng cho mục đích thương mại nếu chưa được phép.\n"
            "Kết quả chỉ mang tính hỗ trợ; chẩn đoán lâm sàng do bác sĩ chuyên khoa quyết định.",
            is_error=False,
        )

    def add_shadow(self, widget):
        # Hàm tạo viền đổ bóng 3D cực xịn cho các Card
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(5)
        shadow.setColor(QColor(0, 0, 0, 30)) # Màu đen với độ trong suốt (alpha) là 30
        widget.setGraphicsEffect(shadow)

    def apply_stylesheet(self):
        # CSS (QSS) thiết kế màu sắc chuẩn Y tế Sáng
        self.setStyleSheet("""
            QMainWindow {
                background-color: #F0F4F8; /* Nền xám xanh nhạt dịu mắt */
            }
            #HeaderTitle {
                font-size: 24px;
                font-weight: 800;
                color: #2C3E50;
                font-family: 'Segoe UI', Arial;
            }
            #HeaderClock {
                font-size: 16px;
                font-weight: bold;
                color: #34495E;
            }
            #CardFrame {
                background-color: #FFFFFF;
                border-radius: 12px;
            }
            #CardTitle {
                font-size: 16px;
                font-weight: bold;
                color: #7F8C8D;
                border-bottom: 2px solid #ECF0F1;
                padding-bottom: 10px;
            }
            #ImageContainer {
                background-color: #F8F9FA;
                border: 2px dashed #BDC3C7;
                border-radius: 8px;
                color: #95A5A6;
                font-size: 14px;
            }
            #ResultLabel {
                font-size: 16px; /* Giảm từ 18px xuống 16px */
                font-weight: bold;
                color: #2C3E50;
                background-color: #F8F9FA;
                padding: 10px; /* Giảm từ 15px xuống 10px để rộng chỗ hơn */
                border-radius: 8px;
            }
            
            /* GIAO DIỆN TỔNG QUÁT CHO CÁC NÚT BẤM */
            QPushButton {
                min-height: 45px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                color: white;
                border: none;
            }

            /* TRẠNG THÁI KHÓA CHUNG: Khi nút chưa được bấm (setEnabled(False)) */
            /* Thêm các ID cụ thể vào đây để đảm bảo chúng chuyển màu xám hoàn toàn */
            QPushButton:disabled, #AiBtn:disabled, #ExportBtn:disabled, #ResetBtn:disabled, #ActionBtn:disabled {
                background-color: #BDC3C7 !important;
                color: #ECF0F1;
            }

            /* MÀU SẮC RIÊNG KHI NÚT ĐƯỢC KÍCH HOẠT */
            #ActionBtn { background-color: #3498DB; } /* Xanh dương - Tải ảnh */
            #ActionBtn:hover { background-color: #2980B9; }
            
            #AiBtn { background-color: #00B894; }     /* Xanh ngọc - Chạy AI */
            #AiBtn:hover { background-color: #009688; }
            
            #ExportBtn { background-color: #9B59B6; } /* Tím - Xuất file */
            #ExportBtn:hover { background-color: #8E44AD; }
            
            #ResetBtn { 
                background-color: #E67E22;           /* Đỏ cam - Reset */
                margin-top: 10px;
            }
            #ResetBtn:hover { background-color: #D35400; }
            
            #AnalysisStatus {
                font-size: 13px;
                font-weight: bold;
                color: #7F8C8D;
                background-color: #FDFEFE;
                border: 1px solid #ECF0F1;
                border-radius: 5px;
                padding: 5px;
                margin-top: 5px;
            }                
                           
            #ResetBtn { 
                background-color: #E67E22; /* Màu cam đỏ */
                margin-top: 10px;
            }
            #ResetBtn:hover { 
                background-color: #D35400; /* Màu đỏ đậm hơn khi di chuột */
            }
            #ResetBtn:disabled {
                background-color: #BDC3C7; /* Màu xám khi chưa có ảnh */
            }
            #SettingBtn { 
                background-color: #ECF0F1; 
                color: #34495E; 
                min-height: 35px;
            }
            #SettingBtn:hover { background-color: #D5D8DC; }
        """
                           )

    def update_clock(self):
        # Hiển thị ngày giờ dạng: 10/04/2026 | 08:30:15
        now = datetime.now().strftime("%d/%m/%Y  |  %H:%M:%S")
        self.lbl_clock.setText(now)

    def show_modern_message(self, title, text, is_error=False):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        
        # 1. Thiết lập biểu tượng
        if is_error:
            msg.setIcon(QMessageBox.Icon.Critical)
            btn_color = "#E74C3C" 
            btn_hover = "#C0392B"
        else:
            msg.setIcon(QMessageBox.Icon.Information)
            btn_color = "#00B894" 
            btn_hover = "#009688"

        # 2. CAN THIỆP LAYOUT: Ép Icon và Text sát nhau về bên trái
        # Chúng ta tìm layout của QMessageBox và chỉnh khoảng cách (spacing)
        layout = msg.layout()
        if layout:
            layout.setSpacing(20) # Khoảng cách giữa Icon và Text chỉ còn 20px
            # Thiết lập để nội dung không bị giãn đều (center) mà dồn về trái
            layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        # 3. CSS TỐI ƯU: Xóa bỏ các min-width gây lỗi nhảy chữ
        msg.setStyleSheet(f"""
            QMessageBox {{
                background-color: #FFFFFF;
                min-width: 350px; /* Độ rộng cố định cho cả hộp thoại */
            }}
            QLabel {{
                color: #2C3E50;
                font-size: 15px;
                font-weight: bold;
                padding: 0px;
                margin: 0px;
                /* Xóa bỏ min-width cũ để chữ tự động bám sát Icon */
                min-width: 0px; 
                qproperty-alignment: 'AlignLeft | AlignVCenter';
            }}
            QPushButton {{
                background-color: {btn_color};
                color: white;
                border-radius: 6px;
                padding: 10px 25px;
                font-weight: bold;
                font-size: 14px;
                min-width: 80px;
                border: none;
                margin-top: 10px;
            }}
            QPushButton:hover {{
                background-color: {btn_hover};
            }}
        """)
        msg.exec()

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh NST",
            "",
            "Ảnh (*.png *.jpg *.jpeg *.tif *.tiff);;Tất cả (*.*)",
        )
        if not file_name:
            return

        img_cv = cv2.imread(file_name)
        if img_cv is None or img_cv.size == 0:
            self.show_modern_message(
                "Ảnh không hợp lệ",
                "Không đọc được file ảnh. Hãy chọn file .jpg, .png hoặc .tif/.tiff còn nguyên vẹn.",
                is_error=True,
            )
            return

        self.current_image_path = file_name
        self.last_result_bgr = None
        self.last_count = None
        self.last_report_plain = ""

        pixmap = QPixmap(file_name).scaled(
            500,
            500,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if pixmap.isNull():
            self.show_modern_message(
                "Ảnh không hợp lệ",
                "Qt không hiển thị được file này. Thử đổi định dạng hoặc kiểm tra file bị hỏng.",
                is_error=True,
            )
            self.current_image_path = None
            return

        self.lbl_image_original.setPixmap(pixmap)
        self.lbl_image_original.setStyleSheet("border: none;")

        self.btn_run_ai.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.btn_save_bundle.setEnabled(False)
        self.lbl_image_result.clear()
        self.lbl_image_result.setText("Kết quả AI phân tích")
        self.lbl_count.setText("Kết quả đếm:\n--")
        self.lbl_status.setText("Báo cáo Y khoa:\n--")

    def reset_all(self):
        # 1. Xóa đường dẫn ảnh và trạng thái bộ nhớ
        self.current_image_path = None
        
        # 2. Xóa ảnh hiển thị (Đưa về trạng thái ban đầu)
        self.lbl_image_original.clear()
        self.lbl_image_original.setText("Ảnh gốc (Chưa tải)")
        self.lbl_image_original.setStyleSheet("") # Khôi phục viền đứt đoạn
        
        self.lbl_image_result.clear()
        self.lbl_image_result.setText("Kết quả AI phân tích")
        self.lbl_image_result.setStyleSheet("")
        
        # 3. Xóa các kết quả văn bản
        self.lbl_count.setText("Kết quả đếm:\n--")
        self.lbl_status.setText("Báo cáo Y khoa:\n--")
        
        # 4. Đưa nhãn trạng thái về ban đầu
        self.lbl_analysis_status.setText("Trạng thái: Sẵn sàng")
        self.lbl_analysis_status.setStyleSheet("") # Khôi phục màu mặc định
        
        # 5. Khóa các nút chức năng
        self.btn_run_ai.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_save_bundle.setEnabled(False)
        self.btn_reset.setEnabled(False) # Khóa chính nó
        self.last_result_bgr = None
        self.last_count = None
        self.last_report_plain = ""

        # (Tùy chọn) Hiện thông báo nhỏ
        self.show_modern_message("Thông báo", "Hệ thống đã được làm mới!")

    def cv_to_qpixmap(self, cv_img, w, h):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, ch = rgb_image.shape
        bytes_per_line = ch * width
        qimg = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)

    def update_animation_text(self):
        # Tạo hiệu ứng: Đang phân tích . -> .. -> ... -> .
        self.dot_count = (self.dot_count + 1) % 4
        dots = "." * self.dot_count
        self.lbl_analysis_status.setText(f"Đang phân tích{dots}")
        self.lbl_analysis_status.setStyleSheet("color: #E67E22; background-color: #FEF5E7;") # Màu cam khi đang chạy

    def run_ai_analysis(self):
        if not self.current_image_path: return

        # 1. Khởi động hiệu ứng và khóa nút
        self.btn_run_ai.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.animation_timer.start(400) # Bắt đầu lượn sóng dấu chấm (0.4s/lần)

        # 2. Sử dụng WorkerThread để chạy ngầm (Tránh treo máy)
        self.worker = WorkerThread(self.ai_analyzer, self.current_image_path)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.start()

    def on_analysis_finished(self, results):
        annotated_bgr, count_result, medical_report_html, medical_report_plain = results

        self.last_result_bgr = annotated_bgr
        self.last_count = count_result
        self.last_report_plain = medical_report_plain

        self.animation_timer.stop()
        self.lbl_analysis_status.setText("Phân tích hoàn tất! ✅")
        self.lbl_analysis_status.setStyleSheet("color: #27AE60; font-weight: bold; background-color: #EAFAF1; padding: 5px; border-radius: 5px;")

        w = self.lbl_image_result.width()
        h = self.lbl_image_result.height()
        self.lbl_image_result.setPixmap(self.cv_to_qpixmap(annotated_bgr, w, h))
        self.lbl_image_result.setStyleSheet("border: none;")

        self.lbl_count.setText(f"Kết quả đếm: {count_result} NST")
        self.lbl_status.setText(medical_report_html)
        self.lbl_status.setStyleSheet("color: #2C3E50; font-size: 15px; background-color: #F8F9FA; padding: 10px; border-radius: 8px; line-height: 1.5;")

        self.btn_run_ai.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_save_bundle.setEnabled(True)

    def on_analysis_failed(self, message: str):
        self.animation_timer.stop()
        self.lbl_analysis_status.setText("Phân tích thất bại")
        self.lbl_analysis_status.setStyleSheet("color: #C0392B; font-weight: bold; background-color: #FADBD8; padding: 5px; border-radius: 5px;")
        self.btn_run_ai.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.show_modern_message("Lỗi phân tích", message, is_error=True)

    def export_csv(self):
        if self.last_count is None:
            QMessageBox.warning(self, "Chưa có kết quả", "Hãy chạy phân tích AI trước khi xuất CSV.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu báo cáo CSV",
            f"Bao_cao_NST_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV (*.csv)",
        )

        if file_path:
            try:
                Exporter.export_to_csv(
                    file_path,
                    self.current_image_path,
                    self.last_count,
                    self.last_report_plain,
                )
                QMessageBox.information(self, "Thành công", f"Đã xuất báo cáo tại:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", f"Chi tiết lỗi:\n{str(e)}")

    def save_result_bundle(self):
        if not self.current_image_path or self.last_result_bgr is None:
            QMessageBox.warning(self, "Chưa có kết quả", "Hãy chạy phân tích AI trước khi lưu bộ kết quả.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu ảnh và báo cáo")
        if not folder:
            return

        try:
            report_path, _ = Exporter.save_result_bundle(
                folder,
                self.current_image_path,
                self.last_result_bgr,
                self.last_count,
                self.last_report_plain,
                int(self.config.get("normal_count", 46)),
                int(self.config.get("tolerance", 1)),
            )
            QMessageBox.information(
                self,
                "Đã lưu",
                f"Đã lưu ảnh và báo cáo trong thư mục:\n{folder}\n\nFile báo cáo: {os.path.basename(report_path)}",
            )
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Không lưu được kết quả:\n{str(e)}")

    def load_config(self):
        self.config_file = "config.yaml"
        defaults = {
            "model_path": "models/best.pt",
            "model_overlap_path": "",
            "model_anomaly_path": "",
            "normal_count": 46,
            "tolerance": 1,
        }
        if not os.path.exists(self.config_file):
            self.config = dict(defaults)
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        else:
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            merged = {**defaults, **loaded}
            self.config = merged

    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.config = dialog.get_config()
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f)
            self.ai_analyzer.config = self.config
            self.ai_analyzer.load_model()
            QMessageBox.information(self, "Thành công", "Đã lưu cấu hình!")