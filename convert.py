import json
import os

# Danh sách 3 file JSON của bạn
json_files = [
    'data/original/4-labels/bounding_box/train_coco.json',
    'data/original/4-labels/bounding_box/val_coco.json',
    'data/original/4-labels/bounding_box/test_coco.json'
]

for file_path in json_files:
    if os.path.exists(file_path):
        # Đọc file cũ
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # 1. Cập nhật phần categories (Chỉ để lại 1 class)
        data['categories'] = [{'id': 1, 'name': 'parasite'}]
        
        # 2. Đổi toàn bộ category_id của annotations về 1
        for ann in data['annotations']:
            ann['category_id'] = 1
            
        # Lưu đè lại file
        with open(file_path, 'w') as f:
            json.dump(data, f)
            
        print(f"✅ Đã chuyển đổi thành công file: {file_path}")
    else:
        print(f"❌ Không tìm thấy file: {file_path}")