import os
import xml.etree.ElementTree as ET
import cv2
from config import config_yolov1 as config


def convert_to_string(image_path, labels):
    out_string = ''
    out_string += image_path
    for label in labels:
        out_string += ' '
        bbox = ",".join(list(map(str, label)))
        out_string += bbox
    out_string += '\n'
    return out_string


def preprocess():
    out_file = open(config.OUTPUT_FILE, 'w', encoding='utf-8')
    xml_dir = os.path.join(config.PASCAL_DIR, config.YEAR, 'Annotations')
    xml_list = os.listdir(xml_dir)

    for xml in xml_list:
        tree = ET.parse(os.path.join(xml_dir, xml))
        image_path = tree.find('filename').text
        image_path = os.path.join(config.PASCAL_DIR, config.YEAR, 'JPEGImages', image_path)
        labels = []
        objs = tree.findall('object')
        for obj in objs:
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            obj_name = obj.find('name').text
            obj_num = config.VOC_CLASS.index(obj_name)
            labels.append([xmin, ymin, xmax, ymax, obj_num])

        print(image_path, labels)
        record = convert_to_string(image_path, labels)
        out_file.write(record)
    out_file.close()

def show_image():
    lines = open(config.OUTPUT_FILE, 'r').read().split('\n')
    for line in lines:
        print(line)
        if len(line) == 0:
            continue
        image_path = line.split(' ')[0]
        infos = line.split(' ')[1:]
        image = cv2.imread(image_path)
        for info in infos:
            temp = info.split(',')
            x1 = int(temp[0].split('.')[0])
            y1 = int(temp[1].split('.')[0])
            x2 = int(temp[2].split('.')[0])
            y2 = int(temp[3].split('.')[0])
            idx = int(temp[4])
            print(temp)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))
            cv2.putText(image, config.VOC_CLASS[idx],
                        (x1, y1 + 20),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (255, 0, 0), thickness=1, lineType=2)
        cv2.imshow('te', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    preprocess()
    show_image()
