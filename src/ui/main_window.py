# src/ui/main_window.py
import os
import yaml
import cv2
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QLabel, QFrame, QStatusBar, QFileDialog, QMessageBox, QDialog,
    QLineEdit, QFormLayout, QSpinBox, QApplication, QGraphicsDropShadowEffect,
    QSizePolicy, QScrollArea, QTabWidget, QProgressBar, QTextBrowser,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage, QIcon, QColor, QAction

from src.core.ai_model import ChromosomeAnalyzer
from src.core.exporter import Exporter


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS DIALOG
# ══════════════════════════════════════════════════════════════════════════════

class SettingsDialog(QDialog):
    def __init__(self, current_config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cài đặt Hệ thống")
        self.setFixedWidth(540)
        self.setStyleSheet("""
            QDialog { background-color: #FFFFFF; }
            QLabel { font-size: 14px; font-weight: bold; color: #34495E; }
            QLineEdit, QSpinBox {
                padding: 10px; border: 2px solid #ECF0F1; border-radius: 8px;
                background-color: #F8F9FA; font-size: 14px; color: #2C3E50;
            }
            QLineEdit:focus, QSpinBox:focus { border: 2px solid #3498DB; background-color: #FFF; }
            QSpinBox::up-button, QSpinBox::down-button { width: 20px; border-radius: 4px; }
            QPushButton { min-height: 40px; min-width: 120px; border-radius: 8px;
                          font-weight: bold; font-size: 14px; border: none; }
        """)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        header = QLabel("CẤU HÌNH HỆ THỐNG")
        header.setStyleSheet("font-size:18px;font-weight:900;color:#2C3E50;"
                             "border-bottom:2px solid #ECF0F1;padding-bottom:10px;")
        main_layout.addWidget(header)

        form = QFormLayout()
        form.setSpacing(12)
        self.model_path_input = QLineEdit(current_config.get("model_path", "models/best.pt"))
        self.model_overlap_input = QLineEdit(current_config.get("model_overlap_path", "") or "")
        self.model_anomaly_input = QLineEdit(current_config.get("model_anomaly_path", "") or "")
        self.normal_count_input = QSpinBox()
        self.normal_count_input.setRange(0, 200)
        self.normal_count_input.setValue(current_config.get("normal_count", 46))
        self.tolerance_input = QSpinBox()
        self.tolerance_input.setRange(0, 50)
        self.tolerance_input.setValue(current_config.get("tolerance", 1))

        form.addRow("Mô hình phân đoạn / đếm NST:", self.model_path_input)
        form.addRow("Mô hình tách NST chồng (tuỳ chọn):", self.model_overlap_input)
        form.addRow("Mô hình hỗ trợ bất thường (tuỳ chọn):", self.model_anomaly_input)
        form.addRow("Số NST chuẩn (2n):", self.normal_count_input)
        form.addRow("Sai số cho phép (±):", self.tolerance_input)
        main_layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = QPushButton("Hủy bỏ")
        btn_cancel.setStyleSheet("QPushButton{background:#ECF0F1;color:#7F8C8D;}"
                                 "QPushButton:hover{background:#BDC3C7;color:white;}")
        btn_cancel.clicked.connect(self.reject)
        btn_save = QPushButton("💾 Lưu cấu hình")
        btn_save.setStyleSheet("QPushButton{background:#00B894;color:white;}"
                               "QPushButton:hover{background:#009688;}")
        btn_save.clicked.connect(self.accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_save)
        main_layout.addLayout(btn_row)

    def get_config(self):
        return {
            "model_path": self.model_path_input.text().strip(),
            "model_overlap_path": self.model_overlap_input.text().strip(),
            "model_anomaly_path": self.model_anomaly_input.text().strip(),
            "normal_count": self.normal_count_input.value(),
            "tolerance": self.tolerance_input.value(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL PANEL — hiển thị bảng Denver + thống kê hình học
# ══════════════════════════════════════════════════════════════════════════════

class DetailPanel(QFrame):
    """Panel Tab bên phải hiển thị kết quả phân tích chi tiết — có con lăn chuột."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CardFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel("PHÂN TÍCH CHI TIẾT")
        title.setObjectName("CardTitle")
        layout.addWidget(title)

        # QTextBrowser: hỗ trợ HTML + cuộn chuột tốt hơn QLabel trong ScrollArea
        self.content = QTextBrowser()
        self.content.setObjectName("DetailContent")
        self.content.setOpenExternalLinks(False)
        self.content.setFrameShape(QFrame.Shape.NoFrame)
        self.content.setStyleSheet("""
            QTextBrowser {
                background: transparent;
                border: none;
                font-size: 13px;
                color: #2C3E50;
            }
            QScrollBar:vertical {
                background: #F0F4F8;
                width: 8px;
                border-radius: 4px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #BDC3C7;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00B894;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)
        self.content.setHtml("<p style='color:#95A5A6;'>Chưa có dữ liệu.<br>Chạy AI để xem kết quả chi tiết.</p>")
        layout.addWidget(self.content)

    def update_report(self, analysis_report, total: int):
        from src.core.chromosome_classifier import AnalysisReport
        r: AnalysisReport = analysis_report

        # ── Giới tính ────────────────────────────────────────────────────
        sex_color = {"Cao": "#27AE60", "Trung bình": "#E67E22", "Thấp": "#E74C3C"}
        sc = sex_color.get(r.sex_confidence, "#7F8C8D")

        # ── Bảng nhóm Denver ─────────────────────────────────────────────
        denver_rows = ""
        group_desc = {
            "A": "NST 1–3 (lớn nhất)",
            "B": "NST 4–5",
            "C": "NST 6–12, X",
            "D": "NST 13–15 (tâm đầu)",
            "E": "NST 16–18",
            "F": "NST 19–20",
            "G": "NST 21–22, Y (nhỏ nhất)",
        }
        for g, n in r.group_sizes.items():
            bar_width = min(int(n * 12), 100)
            bar_color = "#3498DB" if n > 0 else "#ECF0F1"
            denver_rows += f"""
            <tr>
              <td style='padding:3px 8px;font-weight:bold;color:#2C3E50;'>Nhóm {g}</td>
              <td style='padding:3px 6px;color:#7F8C8D;font-size:11px;'>{group_desc.get(g,"")}</td>
              <td style='padding:3px 8px;'>
                <div style='display:inline-block;width:{bar_width}px;height:10px;
                     background:{bar_color};border-radius:5px;'></div>
              </td>
              <td style='padding:3px 6px;font-weight:bold;color:#2C3E50;'>{n}</td>
            </tr>"""

        # ── Thống kê hình học ────────────────────────────────────────────
        stats = r.size_stats
        stats_html = ""
        if stats:
            stats_html = f"""
            <div style='background:#F8F9FA;border-radius:8px;padding:10px;margin-top:6px;'>
              <b style='color:#2C3E50;'>Thống kê kích thước mask</b><br>
              <table style='font-size:13px;margin-top:6px;'>
                <tr><td style='color:#7F8C8D;padding:2px 12px 2px 0;'>Trung bình (chuẩn hoá):</td>
                    <td><b>{stats.get('mean_area',0):.3f}</b></td></tr>
                <tr><td style='color:#7F8C8D;padding:2px 12px 2px 0;'>Độ lệch chuẩn:</td>
                    <td><b>{stats.get('std_area',0):.3f}</b></td></tr>
                <tr><td style='color:#7F8C8D;padding:2px 12px 2px 0;'>Nhỏ nhất / Lớn nhất:</td>
                    <td><b>{stats.get('min_area',0):.3f} / {stats.get('max_area',0):.3f}</b></td></tr>
                <tr><td style='color:#7F8C8D;padding:2px 12px 2px 0;'>Hệ số biến thiên:</td>
                    <td><b>{stats.get('cv_percent',0):.1f}%</b></td></tr>
              </table>
            </div>"""

        # ── Cảnh báo hội chứng ────────────────────────────────────────────
        syndrome_html = ""
        if r.syndrome_flags:
            items = "".join(
                f"<li style='color:#C0392B;margin:4px 0;'>{s}</li>"
                for s in r.syndrome_flags
            )
            syndrome_html = f"""
            <div style='background:#FDEDEC;border-left:4px solid #C0392B;
                 border-radius:4px;padding:10px;margin-top:8px;'>
              <b style='color:#C0392B;'>⚠ Hội chứng nghi ngờ:</b>
              <ul style='margin:6px 0 0 16px;padding:0;'>{items}</ul>
            </div>"""
        else:
            syndrome_html = """
            <div style='background:#EAFAF1;border-left:4px solid #27AE60;
                 border-radius:4px;padding:10px;margin-top:8px;'>
              <span style='color:#27AE60;'>✅ Không phát hiện hội chứng đặc trưng theo số lượng NST</span>
            </div>"""

        html = f"""
        <div style='font-family:Segoe UI,Arial;font-size:14px;line-height:1.7;'>

          <div style='background:#EBF5FB;border-radius:8px;padding:10px;margin-bottom:8px;'>
            <b style='color:#2C3E50;'>🧬 Giới tính ước tính</b><br>
            <span style='font-size:15px;font-weight:bold;'>{r.sex_estimation}</span>
            <span style='color:{sc};font-size:12px;'> — Độ tin cậy: {r.sex_confidence}</span><br>
            <span style='color:#95A5A6;font-size:11px;'>
              Ước tính từ hình học mask. Cần karyotype để xác nhận chính xác.</span>
          </div>

          <b style='color:#2C3E50;'>📊 Phân nhóm Denver</b>
          <table style='width:100%;margin-top:6px;border-collapse:collapse;'>
            {denver_rows}
          </table>

          {stats_html}
          {syndrome_html}

          <div style='margin-top:10px;padding:8px;background:#FDFEFE;border-radius:6px;
               border:1px solid #ECF0F1;'>
            <span style='color:#7F8C8D;font-size:11px;'>
              ℹ Phân nhóm Denver dựa trên kích thước tương đối mask phân đoạn.<br>
              Kết quả hỗ trợ nghiên cứu/giáo dục — không thay thế xét nghiệm lâm sàng.
            </span>
          </div>
        </div>
        """
        self.content.setHtml(html)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Phần mềm Phân tích Nhiễm sắc thể - AI")
        self.setGeometry(100, 100, 1400, 860)
        self.setWindowIcon(QIcon("assets/logoNST.png"))

        self.current_image_path = None
        self.last_result_bgr = None
        self.last_count = None
        self.last_report_plain = ""
        self.last_analysis_report = None

        self.load_config()
        self.ai_analyzer = ChromosomeAnalyzer(self.config)

        self.init_ui()
        self._build_menu_bar()
        self.apply_stylesheet()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.update_clock()

    # ──────────────────────────────────────────────────────────────────────
    # UI BUILD
    # ──────────────────────────────────────────────────────────────────────

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        master_layout = QVBoxLayout(main_widget)
        master_layout.setContentsMargins(20, 20, 20, 20)
        master_layout.setSpacing(16)

        # ── Header ────────────────────────────────────────────────────────
        header_frame = QFrame()
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.lbl_logo = QLabel()
        if os.path.exists("assets/logoNST.png"):
            px = QPixmap("assets/logoNST.png").scaled(
                56, 56, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation)
            self.lbl_logo.setPixmap(px)
        else:
            self.lbl_logo.setText("🏥")
            self.lbl_logo.setFont(QFont("Arial", 28))

        self.lbl_title = QLabel("HỆ THỐNG PHÂN TÍCH NHIỄM SẮC THỂ AI")
        self.lbl_title.setObjectName("HeaderTitle")

        self.lbl_clock = QLabel("00:00:00")
        self.lbl_clock.setObjectName("HeaderClock")
        self.lbl_clock.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        header_layout.addWidget(self.lbl_logo)
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.lbl_clock)
        master_layout.addWidget(header_frame)

        # ── Body ──────────────────────────────────────────────────────────
        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)
        master_layout.addLayout(body_layout)

        # ── Cột 1: Control Panel ─────────────────────────────────────────
        control_panel = QFrame()
        control_panel.setObjectName("CardFrame")
        self.add_shadow(control_panel)
        control_panel.setFixedWidth(240)
        ctrl = QVBoxLayout(control_panel)
        ctrl.setContentsMargins(18, 18, 18, 18)
        ctrl.setSpacing(12)

        lbl_ctrl = QLabel("BẢNG ĐIỀU KHIỂN")
        lbl_ctrl.setObjectName("CardTitle")

        self.btn_load = QPushButton("📂 Tải ảnh lên")
        self.btn_load.setObjectName("ActionBtn")
        self.btn_load.clicked.connect(self.load_image)

        self.btn_run_ai = QPushButton("🧠 Chạy AI Phân tích")
        self.btn_run_ai.setObjectName("AiBtn")
        self.btn_run_ai.setEnabled(False)
        self.btn_run_ai.clicked.connect(self.run_ai_analysis)

        self.lbl_analysis_status = QLabel("Trạng thái: Sẵn sàng")
        self.lbl_analysis_status.setObjectName("AnalysisStatus")
        self.lbl_analysis_status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Progress bar nhỏ
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar{border-radius:3px;background:#ECF0F1;}"
            "QProgressBar::chunk{background:#00B894;border-radius:3px;}"
        )

        self.btn_reset = QPushButton("🔄 Tải lại (Reset)")
        self.btn_reset.setObjectName("ResetBtn")
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self.reset_all)

        self.dot_count = 0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation_text)

        self.btn_settings = QPushButton("⚙️ Cài đặt hệ thống")
        self.btn_settings.setObjectName("SettingBtn")
        self.btn_settings.clicked.connect(self.open_settings)

        ctrl.addWidget(lbl_ctrl)
        ctrl.addWidget(self.btn_load)
        ctrl.addWidget(self.btn_run_ai)
        ctrl.addWidget(self.lbl_analysis_status)
        ctrl.addWidget(self.progress_bar)
        ctrl.addWidget(self.btn_reset)
        ctrl.addStretch()
        ctrl.addWidget(self.btn_settings)

        # ── Cột 2: Ảnh ───────────────────────────────────────────────────
        image_panel = QFrame()
        image_panel.setObjectName("CardFrame")
        self.add_shadow(image_panel)
        img_layout = QHBoxLayout(image_panel)
        img_layout.setContentsMargins(12, 12, 12, 12)
        img_layout.setSpacing(12)

        self.lbl_image_original = QLabel("Ảnh gốc\n(Chưa tải)")
        self.lbl_image_result = QLabel("Kết quả AI\nphân tích")
        for lbl in [self.lbl_image_original, self.lbl_image_result]:
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setObjectName("ImageContainer")
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            img_layout.addWidget(lbl)

        # ── Cột 3: Tab kết quả + chi tiết ───────────────────────────────
        right_panel = QFrame()
        right_panel.setObjectName("CardFrame")
        self.add_shadow(right_panel)
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("ResultTabs")

        # Tab 1: Tóm tắt
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.setContentsMargins(16, 16, 16, 16)
        summary_layout.setSpacing(12)

        lbl_title_result = QLabel("KẾT QUẢ PHÂN TÍCH")
        lbl_title_result.setObjectName("CardTitle")

        self.lbl_count = QLabel("Kết quả đếm: —")
        self.lbl_count.setObjectName("ResultLabel")
        self.lbl_count.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.lbl_count.setWordWrap(True)

        self.lbl_status = QLabel("Báo cáo:\n—")
        self.lbl_status.setObjectName("ResultLabel")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.lbl_status.setTextFormat(Qt.TextFormat.RichText)

        self.btn_export = QPushButton("📥 Xuất báo cáo CSV")
        self.btn_export.setObjectName("ExportBtn")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_csv)

        self.btn_save_bundle = QPushButton("💾 Lưu ảnh & báo cáo")
        self.btn_save_bundle.setObjectName("ExportBtn")
        self.btn_save_bundle.setEnabled(False)
        self.btn_save_bundle.setToolTip("Lưu ảnh gốc, ảnh phân đoạn và file báo cáo .txt")
        self.btn_save_bundle.clicked.connect(self.save_result_bundle)

        summary_layout.addWidget(lbl_title_result)
        summary_layout.addWidget(self.lbl_count)
        summary_layout.addWidget(self.lbl_status, stretch=1)
        summary_layout.addWidget(self.btn_save_bundle)
        summary_layout.addWidget(self.btn_export)

        # Tab 2: Chi tiết
        self.detail_panel = DetailPanel()

        self.tabs.addTab(summary_widget, "📋 Tóm tắt")
        self.tabs.addTab(self.detail_panel, "🔬 Chi tiết")

        right_layout.addWidget(self.tabs)

        # ── Ráp body ────────────────────────────────────────────────────
        body_layout.addWidget(control_panel)
        body_layout.addWidget(image_panel, 1)
        body_layout.addWidget(right_panel)

        # ── Status bar ──────────────────────────────────────────────────
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage(
            "Trình tự: Tải ảnh → Chạy AI → Xem kết quả (Tab Tóm tắt + Chi tiết) → Lưu / Xuất"
        )
        copy_lbl = QLabel("© Đồ án NST-AI — Phần mềm hỗ trợ nghiên cứu / giáo dục")
        copy_lbl.setStyleSheet("color:#7F8C8D;font-size:11px;padding-right:8px;")
        self.statusBar.addPermanentWidget(copy_lbl)

    def _build_menu_bar(self):
        menu = self.menuBar()
        help_menu = menu.addMenu("Trợ giúp")
        act_about = QAction("Giới thiệu & bản quyền", self)
        act_about.triggered.connect(self.show_about_copyright)
        help_menu.addAction(act_about)

    # ──────────────────────────────────────────────────────────────────────
    # STYLESHEET
    # ──────────────────────────────────────────────────────────────────────

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #F0F4F8; }
            #HeaderTitle { font-size:22px;font-weight:800;color:#2C3E50;font-family:'Segoe UI',Arial; }
            #HeaderClock { font-size:15px;font-weight:bold;color:#34495E; }
            #CardFrame { background-color:#FFFFFF;border-radius:12px; }
            #CardTitle { font-size:14px;font-weight:bold;color:#7F8C8D;
                         border-bottom:2px solid #ECF0F1;padding-bottom:8px; }
            #ImageContainer { background-color:#F8F9FA;border:2px dashed #BDC3C7;
                              border-radius:8px;color:#95A5A6;font-size:13px; }
            #ResultLabel { font-size:14px;font-weight:bold;color:#2C3E50;
                           background-color:#F8F9FA;padding:10px;border-radius:8px; }
            #DetailContent { font-size:13px;color:#2C3E50;padding:4px; }

            /* Tabs */
            QTabWidget::pane { border:none;background:#FFFFFF;border-radius:0 0 12px 12px; }
            QTabBar::tab { padding:10px 20px;font-size:13px;font-weight:bold;
                           color:#7F8C8D;background:#F8F9FA;border:none;
                           border-radius:8px 8px 0 0;margin-right:2px; }
            QTabBar::tab:selected { color:#2C3E50;background:#FFFFFF;
                                    border-bottom:3px solid #00B894; }

            QPushButton { min-height:42px;border-radius:6px;font-weight:bold;
                          font-size:13px;color:white;border:none; }
            QPushButton:disabled { background-color:#BDC3C7 !important;color:#ECF0F1; }
            #ActionBtn { background-color:#3498DB; }
            #ActionBtn:hover { background-color:#2980B9; }
            #AiBtn { background-color:#00B894; }
            #AiBtn:hover { background-color:#009688; }
            #ExportBtn { background-color:#9B59B6; }
            #ExportBtn:hover { background-color:#8E44AD; }
            #ResetBtn { background-color:#E67E22;margin-top:6px; }
            #ResetBtn:hover { background-color:#D35400; }
            #ResetBtn:disabled { background-color:#BDC3C7; }
            #SettingBtn { background-color:#ECF0F1;color:#34495E;min-height:34px; }
            #SettingBtn:hover { background-color:#D5D8DC; }
            #AnalysisStatus { font-size:12px;font-weight:bold;color:#7F8C8D;
                              background:#FDFEFE;border:1px solid #ECF0F1;
                              border-radius:5px;padding:5px;margin-top:4px; }
        """)

    def add_shadow(self, widget):
        s = QGraphicsDropShadowEffect()
        s.setBlurRadius(20)
        s.setXOffset(0)
        s.setYOffset(5)
        s.setColor(QColor(0, 0, 0, 28))
        widget.setGraphicsEffect(s)

    # ──────────────────────────────────────────────────────────────────────
    # LOGIC
    # ──────────────────────────────────────────────────────────────────────

    def update_clock(self):
        now = datetime.now().strftime("%d/%m/%Y  |  %H:%M:%S")
        self.lbl_clock.setText(now)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Chọn ảnh NST", "",
            "Ảnh (*.png *.jpg *.jpeg *.tif *.tiff);;Tất cả (*.*)",
        )
        if not file_name:
            return

        img_cv = cv2.imread(file_name)
        if img_cv is None or img_cv.size == 0:
            self.show_modern_message("Ảnh không hợp lệ",
                "Không đọc được file ảnh. Hãy chọn file .jpg, .png hoặc .tif/.tiff còn nguyên vẹn.",
                is_error=True)
            return

        self.current_image_path = file_name
        self.last_result_bgr = None
        self.last_count = None
        self.last_report_plain = ""
        self.last_analysis_report = None

        pixmap = QPixmap(file_name).scaled(
            500, 500, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        if pixmap.isNull():
            self.show_modern_message("Ảnh không hợp lệ",
                "Qt không hiển thị được file này.", is_error=True)
            self.current_image_path = None
            return

        self.lbl_image_original.setPixmap(pixmap)
        self.lbl_image_original.setStyleSheet("border:none;")
        self.btn_run_ai.setEnabled(True)
        self.btn_reset.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.btn_save_bundle.setEnabled(False)
        self.lbl_image_result.clear()
        self.lbl_image_result.setText("Kết quả AI phân tích")
        self.lbl_count.setText("Kết quả đếm: —")
        self.lbl_status.setText("Báo cáo:—")
        self.detail_panel.content.setHtml("<p style='color:#95A5A6;'>Chưa có dữ liệu.<br>Chạy AI để xem kết quả chi tiết.</p>")

    def reset_all(self):
        self.current_image_path = None
        self.lbl_image_original.clear()
        self.lbl_image_original.setText("Ảnh gốc\n(Chưa tải)")
        self.lbl_image_original.setStyleSheet("")
        self.lbl_image_result.clear()
        self.lbl_image_result.setText("Kết quả AI phân tích")
        self.lbl_image_result.setStyleSheet("")
        self.lbl_count.setText("Kết quả đếm: —")
        self.lbl_status.setText("Báo cáo:\n—")
        self.lbl_analysis_status.setText("Trạng thái: Sẵn sàng")
        self.lbl_analysis_status.setStyleSheet("")
        self.detail_panel.content.setHtml("<p style='color:#95A5A6;'>Chưa có dữ liệu.<br>Chạy AI để xem kết quả chi tiết.</p>")
        self.btn_run_ai.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_save_bundle.setEnabled(False)
        self.btn_reset.setEnabled(False)
        self.last_result_bgr = None
        self.last_count = None
        self.last_report_plain = ""
        self.last_analysis_report = None
        self.show_modern_message("Thông báo", "Hệ thống đã được làm mới!")

    def update_animation_text(self):
        self.dot_count = (self.dot_count + 1) % 4
        dots = "." * self.dot_count
        self.lbl_analysis_status.setText(f"Đang phân tích{dots}")
        self.lbl_analysis_status.setStyleSheet(
            "color:#E67E22;background-color:#FEF5E7;")

    def run_ai_analysis(self):
        if not self.current_image_path:
            return
        self.btn_run_ai.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.animation_timer.start(400)
        self.worker = WorkerThread(self.ai_analyzer, self.current_image_path)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.failed.connect(self.on_analysis_failed)
        self.worker.start()

    def on_analysis_finished(self, results):
        annotated_bgr, count_result, report_html, report_plain, analysis_report = results

        self.last_result_bgr = annotated_bgr
        self.last_count = count_result
        self.last_report_plain = report_plain
        self.last_analysis_report = analysis_report

        self.animation_timer.stop()
        self.progress_bar.setVisible(False)
        self.lbl_analysis_status.setText("Phân tích hoàn tất! ✅")
        self.lbl_analysis_status.setStyleSheet(
            "color:#27AE60;font-weight:bold;background-color:#EAFAF1;"
            "padding:5px;border-radius:5px;")

        w = self.lbl_image_result.width()
        h = self.lbl_image_result.height()
        self.lbl_image_result.setPixmap(self.cv_to_qpixmap(annotated_bgr, w, h))
        self.lbl_image_result.setStyleSheet("border:none;")

        self.lbl_count.setText(f"<b style='font-size:20px;'>{count_result}</b> NST đếm được")
        self.lbl_count.setTextFormat(Qt.TextFormat.RichText)
        self.lbl_status.setText(report_html)
        self.lbl_status.setStyleSheet(
            "color:#2C3E50;font-size:13px;background:#F8F9FA;padding:10px;border-radius:8px;")

        # Cập nhật tab chi tiết
        self.detail_panel.update_report(analysis_report, count_result)

        self.btn_run_ai.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_save_bundle.setEnabled(True)

        # Tự động chuyển sang tab Chi tiết nếu phát hiện bất thường
        if not analysis_report.is_normal_count or analysis_report.syndrome_flags:
            self.tabs.setCurrentIndex(1)

    def on_analysis_failed(self, message: str):
        self.animation_timer.stop()
        self.progress_bar.setVisible(False)
        self.lbl_analysis_status.setText("Phân tích thất bại ❌")
        self.lbl_analysis_status.setStyleSheet(
            "color:#C0392B;font-weight:bold;background:#FADBD8;padding:5px;border-radius:5px;")
        self.btn_run_ai.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.show_modern_message("Lỗi phân tích", message, is_error=True)

    def cv_to_qpixmap(self, cv_img, w, h):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, ch = rgb.shape
        qimg = QImage(rgb.data, width, height, ch * width, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg).scaled(
            w, h, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)

    # ──────────────────────────────────────────────────────────────────────
    # EXPORT
    # ──────────────────────────────────────────────────────────────────────

    def export_csv(self):
        if self.last_count is None:
            QMessageBox.warning(self, "Chưa có kết quả", "Hãy chạy phân tích AI trước.")
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Lưu báo cáo CSV",
            f"Bao_cao_NST_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV (*.csv)")
        if file_path:
            try:
                Exporter.export_to_csv(file_path, self.current_image_path,
                                       self.last_count, self.last_report_plain)
                QMessageBox.information(self, "Thành công", f"Đã xuất báo cáo:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Lỗi", str(e))

    def save_result_bundle(self):
        if not self.current_image_path or self.last_result_bgr is None:
            QMessageBox.warning(self, "Chưa có kết quả", "Hãy chạy phân tích AI trước.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Chọn thư mục lưu")
        if not folder:
            return
        try:
            report_path, _ = Exporter.save_result_bundle(
                folder, self.current_image_path, self.last_result_bgr,
                self.last_count, self.last_report_plain,
                int(self.config.get("normal_count", 46)),
                int(self.config.get("tolerance", 1)))
            QMessageBox.information(self, "Đã lưu",
                f"Lưu thành công tại:\n{folder}\nFile báo cáo: {os.path.basename(report_path)}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    # ──────────────────────────────────────────────────────────────────────
    # CONFIG & ABOUT
    # ──────────────────────────────────────────────────────────────────────

    def load_config(self):
        self.config_file = "config.yaml"
        defaults = {"model_path": "models/best.pt", "model_overlap_path": "",
                    "model_anomaly_path": "", "normal_count": 46, "tolerance": 1}
        if not os.path.exists(self.config_file):
            self.config = dict(defaults)
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f, allow_unicode=True, sort_keys=False)
        else:
            with open(self.config_file, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
            self.config = {**defaults, **loaded}

    def open_settings(self):
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            self.config = dialog.get_config()
            with open(self.config_file, "w", encoding="utf-8") as f:
                yaml.dump(self.config, f)
            self.ai_analyzer.config = self.config
            self.ai_analyzer.load_model()
            QMessageBox.information(self, "Thành công", "Đã lưu cấu hình!")

    def show_about_copyright(self):
        self.show_modern_message(
            "Giới thiệu & bản quyền",
            "Phần mềm phân tích NST tích hợp mô hình AI — đồ án môn học.\n\n"
            "Tính năng: Phân đoạn, đếm NST · Ước tính giới tính (XX/XY) · "
            "Phân nhóm Denver · Phát hiện hội chứng di truyền (rule-based).\n\n"
            "© Bản quyền thuộc tác giả dự án. "
            "Kết quả chỉ mang tính hỗ trợ; chẩn đoán lâm sàng do bác sĩ chuyên khoa quyết định.",
            is_error=False)

    def show_modern_message(self, title, text, is_error=False):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(text)
        if is_error:
            msg.setIcon(QMessageBox.Icon.Critical)
            btn_color, btn_hover = "#E74C3C", "#C0392B"
        else:
            msg.setIcon(QMessageBox.Icon.Information)
            btn_color, btn_hover = "#00B894", "#009688"
        msg.setStyleSheet(f"""
            QMessageBox {{ background:#FFFFFF;min-width:350px; }}
            QLabel {{ color:#2C3E50;font-size:14px;font-weight:bold;min-width:0px; }}
            QPushButton {{ background:{btn_color};color:white;border-radius:6px;
                           padding:10px 25px;font-weight:bold;font-size:14px;
                           min-width:80px;border:none;margin-top:10px; }}
            QPushButton:hover {{ background:{btn_hover}; }}
        """)
        msg.exec()