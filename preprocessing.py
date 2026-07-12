import numpy as np
import cv2
from utils import visualize_image, get_filenames
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img

if __name__ == "__main__":
    filenames_list = get_filenames()
    visualize_image(filename=filenames_list[0], clahe=True)