# src/main.py
import sys
import os
import ctypes # <--- Thư viện dùng để can thiệp vào taskbar Windows

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

if __name__ == "__main__":
    # --- ĐOẠN CODE FIX LỖI ICON TASKBAR ---
    try:
        # Tạo một mã định danh (AppID) riêng cho app của bạn
        myappid = 'hospital.chromosome_analyzer.version1' 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass # Nếu chạy trên Mac/Linux thì tự động bỏ qua
    # --------------------------------------

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.showMaximized() 
    
    sys.exit(app.exec())