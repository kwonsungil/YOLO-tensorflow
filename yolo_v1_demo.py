from model.yolo_v1 import yolo_v1
import cv2
import os

if __name__ == '__main__':

    yolo = yolo_v1(False)
    sample_dir = 'D:\\yolo_keras\\sample'
    images = os.listdir(sample_dir)
    for image in images:
        image = cv2.imread(os.path.join(sample_dir, image))
        result = yolo.detect(image)
        yolo.draw_result(image, result)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
