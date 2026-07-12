import json
import os
from collections import defaultdict

# 1. Cấu hình đường dẫn
# Sửa lại đường dẫn này cho khớp với vị trí thư mục chứa 3 file json của bạn
folder_path = r"data/original/4-labels/bounding_box" 
files = ["train_coco.json", "val_coco.json", "test_coco.json"]

# Dictionary để lưu thống kê: {category_name: {'train': 0, 'val': 0, 'test': 0, 'total': 0}}
stats = defaultdict(lambda: {"train": 0, "val": 0, "test": 0, "total": 0})
category_map = {} # Ánh xạ category_id -> category_name

# 2. Xử lý từng file
for file_name in files:
    file_path = os.path.join(folder_path, file_name)
    split_name = file_name.split('_')[0] # Lấy ra chữ 'train', 'val', hoặc 'test'
    
    if not os.path.exists(file_path):
        print(f"Cảnh báo: Không tìm thấy file {file_path}")
        continue
        
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # Cập nhật danh sách tên nhãn từ trường 'categories'
        for cat in data.get('categories', []):
            category_map[cat['id']] = cat['name']
            
        # Thống kê từ trường 'annotations'
        for ann in data.get('annotations', []):
            cat_id = ann['category_id']
            # Nếu vì lý do nào đó id không có trong map, đặt tên là Unknown
            cat_name = category_map.get(cat_id, f"Unknown_{cat_id}") 
            
            stats[cat_name][split_name] += 1
            stats[cat_name]["total"] += 1

# 3. In kết quả ra màn hình dưới dạng bảng
print(f"{'Tên Nhãn (Class)':<20} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'TỔNG CỘNG':<8}")
print("-" * 65)

total_train = total_val = total_test = total_all = 0

# Sắp xếp theo tên nhãn để dễ nhìn
for cat_name in sorted(stats.keys()):
    counts = stats[cat_name]
    print(f"{cat_name:<20} | {counts['train']:<8} | {counts['val']:<8} | {counts['test']:<8} | {counts['total']:<8}")
    
    # Cộng dồn tổng toàn bộ dataset
    total_train += counts['train']
    total_val += counts['val']
    total_test += counts['test']
    total_all += counts['total']

print("=" * 65)
print(f"{'TỔNG TOÀN BỘ DATASET':<20} | {total_train:<8} | {total_val:<8} | {total_test:<8} | {total_all:<8}")