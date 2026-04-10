import os
import shutil

def prepare_cirnet_data():
    # 1. Đường dẫn thư mục gốc CIR-Net của bạn
    origin_dir = r"C:\Users\khang\OneDrive\Documents\NST_Data\CIR-Net-master\data\origin"
    
    # 2. Nơi chứa data đã phân loại để chuẩn bị train AI
    output_dir = r"C:\Users\khang\OneDrive\Documents\NST_AI_Project\dataset_classifier"

    # Tạo sẵn 24 thư mục (ch1 -> ch22, chx, chy)
    classes = [f"ch{i}" for i in range(1, 23)] + ["chx", "chy"]
    for cls in classes:
        os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

    print("Đang xử lý và di chuyển ảnh...")
    success_count = 0

    # Lặp qua toàn bộ file .tiff
    for filename in os.listdir(origin_dir):
        if filename.endswith(".tiff"):
            try:
                # Tách lấy phần tên trước đuôi .tiff
                name_without_ext = filename.rsplit(".", 1)[0]
                # Lấy con số cuối cùng sau dấu chấm
                label_str = name_without_ext.split(".")[-1]
                label_int = int(label_str)
                
                # Xác định thư mục đích
                if 1 <= label_int <= 22:
                    folder_name = f"ch{label_int}"
                elif label_int == 23:
                    folder_name = "chx"
                elif label_int == 24:
                    folder_name = "chy"
                else:
                    continue # Bỏ qua nếu có nhãn lạ
                    
                # Copy ảnh vào thư mục tương ứng
                src_path = os.path.join(origin_dir, filename)
                dst_path = os.path.join(output_dir, folder_name, filename)
                shutil.copy2(src_path, dst_path)
                success_count += 1
                
            except Exception as e:
                print(f"Lỗi đọc file {filename}: {e}")

    print(f"✅ Hoàn tất! Đã phân loại thành công {success_count} ảnh vào thư mục {output_dir}")

if __name__ == "__main__":
    prepare_cirnet_data()