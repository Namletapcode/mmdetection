import re
import matplotlib.pyplot as plt

# 1. Đường dẫn tới file text của bạn
log_file = 'download.txt'

# 2. Khởi tạo danh sách lưu dữ liệu
train_iters = []
train_metrics = {
    'loss': [], 'loss_rpn_cls': [], 'loss_rpn_bbox': [], 
    'loss_cls': [], 'loss_bbox': [], 'acc': []
}

val_epochs = []
val_metrics = {
    'bbox_mAP': [], 'bbox_mAP_50': [], 'bbox_mAP_75': [], 
    'bbox_mAP_m': [], 'bbox_mAP_l': []
}

# Hàm phụ trợ để tìm con số đứng sau từ khóa
def extract_value(keyword, text):
    # Tìm keyword, theo sau là dấu hai chấm, khoảng trắng và bắt lấy con số (có thể có dấu âm)
    match = re.search(rf'{keyword}:\s*(-?[\d.]+)', text)
    if match:
        val = float(match.group(1))
        # Nếu MMDetection trả về -1.000 (tức là không có dữ liệu để chấm), ta đổi thành None để biểu đồ không bị rớt xuống đáy
        return None if val == -1.0 else val
    return None

# 3. Đọc và quét từng dòng trong file TXT
iter_count = 0
epoch_count = 0

with open(log_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    # Nếu dòng chữ chứa chữ 'loss:' và 'loss_cls:' -> Đây là dòng log của bước Train
    if 'loss:' in line and 'loss_cls:' in line:
        iter_count += 1
        train_iters.append(iter_count)
        
        train_metrics['loss'].append(extract_value('loss', line))
        train_metrics['loss_rpn_cls'].append(extract_value('loss_rpn_cls', line))
        train_metrics['loss_rpn_bbox'].append(extract_value('loss_rpn_bbox', line))
        train_metrics['loss_cls'].append(extract_value('loss_cls', line))
        train_metrics['loss_bbox'].append(extract_value('loss_bbox', line))
        train_metrics['acc'].append(extract_value('acc', line))
        
    # Nếu dòng chữ chứa chữ 'coco/bbox_mAP:' -> Đây là dòng log của bước Validate
    elif 'coco/bbox_mAP:' in line:
        epoch_count += 1
        val_epochs.append(epoch_count)
        
        val_metrics['bbox_mAP'].append(extract_value('coco/bbox_mAP', line))
        val_metrics['bbox_mAP_50'].append(extract_value('coco/bbox_mAP_50', line))
        val_metrics['bbox_mAP_75'].append(extract_value('coco/bbox_mAP_75', line))
        val_metrics['bbox_mAP_m'].append(extract_value('coco/bbox_mAP_m', line))
        val_metrics['bbox_mAP_l'].append(extract_value('coco/bbox_mAP_l', line))

# 4. Vẽ biểu đồ Dashboard
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Training Dashboard - Malaria Parasite Detection', fontsize=18, fontweight='bold')

# --- Biểu đồ 1: Tổng Loss (Total Loss) ---
axs[0, 0].plot(train_iters, train_metrics['loss'], color='red', label='Total Loss')
axs[0, 0].set_title('Tổng Loss (Total Loss)', fontsize=14)
axs[0, 0].set_xlabel('Iterations')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].grid(True, linestyle='--', alpha=0.7)
axs[0, 0].legend()

# --- Biểu đồ 2: Các thành phần Loss chi tiết ---
axs[0, 1].plot(train_iters, train_metrics['loss_cls'], label='loss_cls', alpha=0.8)
axs[0, 1].plot(train_iters, train_metrics['loss_bbox'], label='loss_bbox', alpha=0.8)
axs[0, 1].plot(train_iters, train_metrics['loss_rpn_cls'], label='loss_rpn_cls', alpha=0.8)
axs[0, 1].plot(train_iters, train_metrics['loss_rpn_bbox'], label='loss_rpn_bbox', alpha=0.8)
axs[0, 1].set_title('Chi tiết các loại Loss', fontsize=14)
axs[0, 1].set_xlabel('Iterations')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].grid(True, linestyle='--', alpha=0.7)
axs[0, 1].legend()

# --- Biểu đồ 3: Các chỉ số mAP chung ---
if val_epochs:
    axs[1, 0].plot(val_epochs, val_metrics['bbox_mAP'], marker='o', linewidth=2, label='mAP (IoU=0.50:0.95)')
    axs[1, 0].plot(val_epochs, val_metrics['bbox_mAP_50'], marker='s', linewidth=2, label='mAP_50 (AP @ IoU=0.50)')
    axs[1, 0].plot(val_epochs, val_metrics['bbox_mAP_75'], marker='^', linewidth=2, label='mAP_75 (AP @ IoU=0.75)')
    axs[1, 0].set_title('Độ chính xác trung bình (mAP)', fontsize=14)
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('Score')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].legend()

# --- Biểu đồ 4: mAP theo kích thước (Medium & Large) ---
if val_epochs:
    # Đoạn code lọc bỏ các giá trị None (tương đương -1.0 đã bị loại ở trên) để vẽ nét đứt (nếu có đoạn đứt)
    axs[1, 1].plot(val_epochs, val_metrics['bbox_mAP_m'], marker='o', color='purple', label='mAP_m (Vật thể Vừa)')
    axs[1, 1].plot(val_epochs, val_metrics['bbox_mAP_l'], marker='s', color='orange', label='mAP_l (Vật thể Lớn)')
    axs[1, 1].set_title('mAP theo kích thước Ký sinh trùng', fontsize=14)
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('Score')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.92)

# 5. Lưu ảnh chất lượng cao
plt.savefig('malaria_training_dashboard.png', dpi=300)
print("✅ Đã vẽ và lưu biểu đồ thành công: 'malaria_training_dashboard.png'")

plt.show()