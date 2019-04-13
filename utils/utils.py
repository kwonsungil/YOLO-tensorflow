import cv2
import os
import numpy as np
from config import config_yolo_v1 as config


def prepare():
    gt_images = []
    gt_labels = []
    input_np = config.NUMPY_FILE
    if not os.path.exists(input_np):
        lines = open(config.OUTPUT_FILE, 'r', encoding='utf-8').read().split('\n')
        for line in lines:
            if len(line) < 1:
                continue

            # 아무것도 없는 애들 방지
            if line.find(',') == -1:
                continue

            temp = line.split(' ')
            image_path = temp[0]
            image = cv2.imread(image_path)

            h_ratio = 1.0 * config.IMAGE_SIZE / image.shape[0]
            w_ratio = 1.0 * config.IMAGE_SIZE / image.shape[1]
            temp = temp[1:]
            confirm_bbox = []
            # bbox 위치
            label = np.zeros((7, 7, 20 + 5), dtype=np.float32)
            for bbox in temp:
                xmin, ymin, xmax, ymax, class_idx = bbox.split(',')
                xmin = max(min(float(xmin) * w_ratio, config.IMAGE_SIZE), 0)
                ymin = max(min(float(ymin) * h_ratio, config.IMAGE_SIZE), 0)
                xmax = max(min(float(xmax) * w_ratio, config.IMAGE_SIZE), 0)
                ymax = max(min(float(ymax) * h_ratio, config.IMAGE_SIZE), 0)
                # xmin = float(xmin) * w_ratio
                # ymin = float(ymin) * h_ratio
                # xmax = float(xmax) * w_ratio
                # ymax = float(ymax) * h_ratio

                confirm_bbox.append([xmin, ymin, xmax, ymax])

                boxes = [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin]
                gt_x = int(boxes[0] * config.CELL_SIZE / config.IMAGE_SIZE)
                gt_y = int(boxes[1] * config.CELL_SIZE / config.IMAGE_SIZE)
                # 겹치는거에 데해서 처리를 못함
                if label[gt_y, gt_x, 0] == 1:
                    continue
                label[gt_y, gt_x, 0] = 1
                label[gt_y, gt_x, 1:5] = boxes
                # 두 번째 박스 예측은?
                label[gt_y, gt_x, 5 + int(class_idx)] = 1
            # gt_labels.append({'imname': image_path, 'label': label, 'flipped': False})

            image = cv2.resize(image[:, :, (2, 1, 0)], (config.IMAGE_SIZE, config.IMAGE_SIZE))
            for cb in confirm_bbox:
                xmin = cb[0]
                ymin = cb[1]
                xmax = cb[2]
                ymax = cb[3]
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 1)
            # cv2.imshow('adf', image)
            # cv2.waitKey(0)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            # image = (image / 255.0) * 2.0 - 1.0
            gt_images.append(image.astype('uint8'))
            # gt_image_path.append(image_path)
            gt_labels.append(label)
        gt_images = np.array(gt_images, dtype=np.uint8)
        gt_labels = np.array(gt_labels)
        np.savez(input_np, gt_images=gt_images, gt_labels=gt_labels)
    else:
        print('loading dataset...')
        data = np.load(input_np)
        gt_images = data['gt_images']
        gt_labels = data['gt_labels']
        # print(gt_labels[3, :, 1:5])

    return gt_images, gt_labels


def make_batches(gt_images, gt_labels):
    num_batches_per_epoch = int(len(gt_images) / config.BATCH_SIZE) + 1
    # num_batches_per_epoch = int(len(gt_images) / config.BATCH_SIZE)
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * config.BATCH_SIZE
        end_index = min((batch_num + 1) * config.BATCH_SIZE, len(gt_images))
        # yield (gt_images[start_index:end_index, :, :] / 255.0) * 2.0 - 1.0, gt_labels[start_index:end_index, :, :]
        yield (gt_images[start_index:end_index, :, :] / 255), gt_labels[start_index:end_index, :, :]
