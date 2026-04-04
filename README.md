# Malaria Parasite Detection

Dự án này sử dụng framework MMDetection để nhận diện các giai đoạn phát triển của tế bào ký sinh trùng sốt rét.

## 1. Cài đặt môi trường

Chạy lệnh sau tại thư mục gốc của dự án để cài đặt các gói phụ thuộc:

```
pip install -r requirements.txt
```

## 2. Cài đặt data
 - Download checkpoints và images trong link sau: https://drive.google.com/drive/folders/1tGy_f1FR3H6GDhbqQv40XyUdd6J7bIIJ?usp=sharing
 - Thêm thư mục checkpoints vào thư mục custom_model và thư mục images vào thư mục data:
```
data                      
 ┣ images
 ┣ test_coco.json
 ┣ train_coco.json
 ┗ val_coco.json
```
```
custom_model
 ┣ checkpoints
 ┃ ┣ cascade_epoch_22.pth
 ┃ ┣ faster_epoch_22.pth
 ┃ ┗ libra_epoch_24.pth
 ┣ configs
 ┣ custom_cascade.py
 ┣ custom_faster.py
 ┗ custom_libra.py
```

## 3. Trực quan hoá
Chạy lệnh sau để vẽ bounding box trên các ảnh trong tập test (điều chỉnh model và checkpoint phù hợp):
```
python tools/test.py custom_model/custom_cascade.py custom_model/checkpoints/cascade_epoch_22.pth
```

## 4. Đánh giá mô hình
Chạy lệnh sau để đánh giá mô hình (điều chỉnh mô hình cần đánh giá trong tools/predict.py):
```
python tools/predict.py
```
