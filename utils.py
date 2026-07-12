import os
from dotenv import load_dotenv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
load_dotenv()
LABELS_DIR = os.getenv("LABELS_DIR")
IMAGES_DIR = os.getenv("IMAGES_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
RAW_DATASET_DIR = os.getenv("RAW_DATASET_DIR")

COLORS = [(r/255, g/255, b/255) for (r, g, b) in [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128),
    (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (64, 64, 64), (192, 192, 192), (255, 128, 0), (0, 128, 255)
]]
def get_filenames():
    files = os.listdir(IMAGES_DIR)
    filename_list = [os.path.splitext(f)[0] for f in files if os.path.isfile(os.path.join(IMAGES_DIR, f))]
    return sorted(filename_list)

#Trả về list tên các file ảnh và file label đã được sắp xếp 
def images_labels_list():
    return sorted(os.listdir(IMAGES_DIR)), sorted(os.listdir(LABELS_DIR))

#Chuyển về dang bbox (x_center, y_center, width, height) đã được chuẩn hóa về [0, 1]
def convert_to_bboxes_v4(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
    bboxes = []
    class_ids = []
    for line in lines:
        parts = list(map(float, line.strip().split()))
        class_id = int(parts[0])
        coords = parts[1:]
        xs = np.array(coords[0::2])
        ys = np.array(coords[1::2])
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        width = x_max - x_min
        height = y_max - y_min
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        bboxes.append([x_center, y_center, width, height])
        class_ids.append(class_id)
    return bboxes, class_ids

"""
Chuyển từ hệ màu BGR sang LAB(ánh sáng, xanh lục - đổ, xanh lam - vàng), 
áp dụng CLAHE lên kênh L để cải thiện độ tương phản, sau đó chuyển ngược lại về BGR.
"""
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img


def visualize_image(filename, clahe=False):
    def draw_boxes(axis, img, boxes, c_ids, title):
        axis.imshow(img)
        axis.set_title(title)
        axis.axis('off')
        h_img, w_img = img.shape[:2]
        for box, c_id in zip(boxes, c_ids):
            x_c, y_c, w, h = box
            width = w * w_img
            height = h * h_img
            x_min = (x_c * w_img) - (width / 2)
            y_min = (y_c * h_img) - (height / 2)
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=0.5, edgecolor=COLORS[int(c_id)], facecolor='None'
            )
            axis.add_patch(rect)
            axis.text(x_min, y_min - 5, f"ID: {int(c_id)}", 
                      color=COLORS[int(c_id)], fontsize=10, fontweight='bold',
                      bbox=dict(facecolor='red', alpha=0.5, pad=0))
    image_path = os.path.join(IMAGES_DIR, filename + ".jpg")
    label_path = os.path.join(LABELS_DIR, filename + ".txt")
    image = cv2.imread(image_path)
    if clahe:
        image = apply_clahe(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes, class_ids = convert_to_bboxes_v4(label_path)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    draw_boxes(ax, image, bboxes, class_ids, "Ảnh")
    plt.tight_layout()
    plt.show()

def extract_rbc_test(image_path, clahe=False):
    img = cv2.imread(image_path)
    if clahe:
        img = apply_clahe(img)
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=36,
        maxRadius=80
    )
    cells = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0]:
            cells.append((x, y, r))
            cv2.circle(output, (x, y), r, (0, 255, 0), 2)
            cv2.circle(output, (x, y), 2, (0, 255, 0), 2)
    return cells, img, gray, output

def extract_rbc(image_path, clahe=False):
    img = cv2.imread(image_path)
    if clahe:
        img = apply_clahe(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=36,
        maxRadius=80
    )
    cells = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0]:
            cells.append((x, y, r))
    return cells

#Dùng thuật toán Hough Circle Transform để phát hiện các tế bào hồng cầu trong ảnh, 
# trả về tọa độ tâm và bán kính của mỗi tế bào được phát hiện.
#kèm theo ảnh gốc, ảnh xám và ảnh đã được vẽ viền xanh lá lên các hồng cầu để kiểm tra
def show_extracted_rbc(img, gray, output, clahe=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale")
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.imshow(output)
    plt.title("RBC extracted")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

#Tương tự hàm trên nhưng chỉ trả về toạ độ các tế bào hồng cầu mà không hiển thị ảnh
def convert_coods2int(bboxes, width, height):
    x_center, y_center, w, h = bboxes

    x_min = int((x_center - w / 2) * width) + 1
    x_max = int((x_center + w / 2) * width) + 1
    y_min = int((y_center - h / 2) * height) + 1
    y_max = int((y_center + h / 2) * height) + 1
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width, x_max)
    y_max = min(height, y_max)

    return x_min, y_min, x_max, y_max

def compute_iou(bboxes1, bboxes2):
    xcen1, ycen1, w1, h1 = bboxes1
    xcen2, ycen2, w2, h2 = bboxes2
    xmin = max(max(0, int(xcen1 - w1 / 2)), max(0, int(xcen2 - w2 / 2)))
    ymin = max(max(0, int(ycen1 - h1 / 2)), max(0, int(ycen2 - h2 / 2)))
    xmin = min(min(w1, int(xcen1 + w1 / 2)), min(w2, int(xcen2 + w2 / 2)))
    ymin = min(min(h1, int(ycen1 + h1 / 2)), min(h2, int(ycen2 + h2 / 2)))



if __name__ == "__main__":
    print(OUTPUT_DIR)
    images_list, labels_list = images_labels_list()
    filenames_list = get_filenames()
    assert(len(images_list) == len(labels_list))
    print(len(filenames_list))
    bboxes, class_ids = convert_to_bboxes_v4(os.path.join(LABELS_DIR, labels_list[0]))
    assert(len(bboxes) == len(class_ids))
    print(len(bboxes), len(class_ids))
    # visualize_image(filenames_list[118], True)
    cells = extract_rbc(os.path.join(IMAGES_DIR, images_list[36]), False)
    print(f"Number of RBC extracted {len(cells)}")
    cells = extract_rbc(os.path.join(IMAGES_DIR, images_list[36]), True)
    print(f"Number of RBC extracted {len(cells)}")