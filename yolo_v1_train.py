from model.yolo_v1 import yolo_v1
import cv2
from utils import utils
from config import config_yolo_v1 as config
from sklearn.utils import shuffle

if __name__ == '__main__':
    yolo = yolo_v1(True)
    gt_images, gt_labels = utils.prepare()
    print(len(gt_labels))
    print(gt_images.shape)
    print(gt_labels.shape)
    # for step in range(0, self.max_step + 1):
    for epoch in range(config.EPOCHS):
        print('shuffle!!!!')
        gt_images, gt_labels = shuffle(gt_images, gt_labels)
        batches = utils.make_batches(gt_images, gt_labels)
        for batch_idx, batch in enumerate(batches):
            result = yolo.train(epoch, batch_idx, batch)
            print(result)
