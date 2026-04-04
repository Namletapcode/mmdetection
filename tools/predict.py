import os
import cv2
import json
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

config_file = 'custom_model/custom_cascade.py'
checkpoint_file = 'custom_model/checkpoints/cascade_epoch_22.pth'
test_json_path = 'data/test_coco.json'  
images_dir = 'data/images'      
output_dir = 'results'  

os.makedirs(output_dir, exist_ok=True)

device = 'cuda:0' 
model = init_detector(config_file, checkpoint_file, device=device)

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

with open(test_json_path, 'r') as f:
    coco_data = json.load(f)

img_paths = [os.path.join(images_dir, img_info['file_name']) for img_info in coco_data['images']]


for img_path in tqdm(img_paths, desc="Predicting"):
    img_name = os.path.basename(img_path)
    output_path = os.path.join(output_dir, img_name)

    img = cv2.imread(img_path)

    result = inference_detector(model, img) #

    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
        out_file=output_path,
        pred_score_thr=0.5
    )

print(f"\n✅ Hoàn thành! Đã lưu ảnh dự đoán tại thư mục: '{output_dir}'")