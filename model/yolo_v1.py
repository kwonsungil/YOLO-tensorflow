from config import config_yolo_v1 as config
import tensorflow as tf
import datetime
import cv2
import colorsys
import os
import numpy as np


class yolo_v1(object):
    def __init__(self, training=True):
        ######################
        self.classes = config.VOC_CLASS
        self.cell_size = config.CELL_SIZE
        self.image_size = config.IMAGE_SIZE
        self.boxes_per_cell = config.BOXES_PER_CELL
        self.object_scale = config.OBJECT_SCALE
        self.noobject_scale = config.NOOBJECT_SCALE
        self.class_scale = config.CLASS_SCALE
        self.coord_scale = config.COORD_SCALE
        self.batch_size = config.BATCH_SIZE
        self.max_step = config.MAX_STEP
        self.initial_learning_rate = config.LEARNING_RATE
        self.decay_steps = config.DECAY_STEPS
        self.decay_rate = config.DECAY_RATE
        self.staircase = config.STAIRCASE
        self.summary_step = config.SUMMARY_STEP
        self.save_step = config.SAVE_STEP
        self.output_dir = config.OUTPUT_DIR
        self.alpha = config.ALPHA
        self.keep_prob = config.KEEP_PROB
        self.display = True
        self.num_class = len(self.classes)
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell
        self.training = training
        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        ######################
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_network(self.images, training=self.training)
        self.saver = tf.train.Saver()
        # config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
        # gpu_options.allow_growth = True
        gpu_options = tf.GPUOptions()
        gpu_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=gpu_config)

        if training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.learning_rate = tf.train.exponential_decay(
                self.initial_learning_rate, self.global_step, self.decay_steps,
                self.decay_rate, self.staircase, name='learning_rate')
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.compute_loss(self.logits, self.labels)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(
                self.total_loss, global_step=self.global_step)
            tf.summary.scalar('learning_rate', self.learning_rate)
            self.summary_op = tf.summary.merge_all()
            self.logs_dir = os.path.join(config.LOGS_DIR, datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S'))
            os.makedirs(self.logs_dir, exist_ok=True)
            self.writer = tf.summary.FileWriter(self.logs_dir)
            self.writer.add_graph(self.sess.graph)
            filename = os.path.join(config.OUTPUT_DIR, config.WEIGHTS)
            self.sess.run(tf.global_variables_initializer())
            self.save_cfg()
        else:
            self.threshold = config.THRESHOLD
            self.iou_threshold = config.IOU_THRESHOLD
            self.logs_dir = os.path.join(config.LOGS_DIR, os.listdir(config.LOGS_DIR).pop())
            filename = tf.train.latest_checkpoint(self.logs_dir)
            # filename = os.path.join(config.OUTPUT_DIR, config.WEIGHTS)

        if filename is not None:
            print('restore from : ', filename)
            self.saver.restore(self.sess, filename)

    def conv_layer(self, idx, inputs, filters, size, stride, trainable=False):
        weight = tf.Variable(tf.truncated_normal([size, size, int(inputs.shape[3]), filters], stddev=0.1), trainable=trainable)
        biases = tf.Variable(tf.constant(0.1, shape=[filters]), trainable=trainable)
        conv = tf.nn.conv2d(inputs, weight, strides=[1, stride, stride, 1], padding='SAME' if stride == 1 else 'VALID',
                            name=str(idx) + '_conv')
        conv_biased = tf.add(conv, biases, name=str(idx) + '_conv_biased')
        lrelu_conv_b = tf.nn.leaky_relu(conv_biased, self.alpha, name=str(idx) + '_leaky_relu')
        if self.display:
            size1 = lrelu_conv_b.shape[1]
            size2 = lrelu_conv_b.shape[2]
            size3 = lrelu_conv_b.shape[3]
            print('Conv\t%d\t| %d * %d * %d, Stride = %d | %d * %d * %d' % (idx, size, size, filters, stride, size1, size2, size3))
        return lrelu_conv_b

    def pooling_layer(self, idx, inputs, size, stride):
        max_pool = tf.nn.max_pool(inputs, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME', name=str(idx) + '_pool')
        if self.display:
            size1 = max_pool.shape[1]
            size2 = max_pool.shape[2]
            size3 = max_pool.shape[3]
            print('Pool\t%d\t| %d * %d, Stride = %d | %d * %d * %d' % (idx, size, size, stride, size1, size2, size3))
        return max_pool

    def fc_layer(self, idx, inputs, hiddens, flat=False, linear=False, trainable=False):
        weight = tf.Variable(tf.zeros((int(inputs.shape[1]), hiddens)), trainable=trainable)
        biases = tf.Variable(tf.constant(0.1, shape=[hiddens]), trainable=trainable)
        if self.display: print(
            'FC \t%d\t| Input = %d, Output = %d' % (idx, int(inputs.shape[1]), hiddens))
        if linear: return tf.add(tf.matmul(inputs, weight), biases, name=str(idx) + '_fc')
        ip = tf.add(tf.matmul(inputs, weight), biases)
        return tf.maximum(tf.multiply(self.alpha, ip), ip, name=str(idx) + '_fc')

    def dropout(self, inputs, training):
        if training:
            return tf.nn.dropout(inputs, keep_prob=self.keep_prob)
        else:
            return tf.nn.dropout(inputs, keep_prob=1.0)

    def build_network(self, inputs, training=True):
        pad = tf.pad(inputs, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
        conv_1 = self.conv_layer(1, pad, 64, 7, 2)
        pool_2 = self.pooling_layer(2, conv_1, 2, 2)
        conv_3 = self.conv_layer(3, pool_2, 192, 3, 1)
        pool_4 = self.pooling_layer(4, conv_3, 2, 2)
        conv_5 = self.conv_layer(5, pool_4, 128, 1, 1)
        conv_6 = self.conv_layer(6, conv_5, 256, 3, 1)
        conv_7 = self.conv_layer(7, conv_6, 256, 1, 1)
        conv_8 = self.conv_layer(8, conv_7, 512, 3, 1)
        pool_9 = self.pooling_layer(9, conv_8, 2, 2)
        conv_10 = self.conv_layer(10, pool_9, 256, 1, 1)
        conv_11 = self.conv_layer(11, conv_10, 512, 3, 1)
        conv_12 = self.conv_layer(12, conv_11, 256, 1, 1)
        conv_13 = self.conv_layer(13, conv_12, 512, 3, 1)
        conv_14 = self.conv_layer(14, conv_13, 256, 1, 1)
        conv_15 = self.conv_layer(15, conv_14, 512, 3, 1)
        conv_16 = self.conv_layer(16, conv_15, 256, 1, 1)
        conv_17 = self.conv_layer(17, conv_16, 512, 3, 1)
        conv_18 = self.conv_layer(18, conv_17, 512, 1, 1)
        conv_19 = self.conv_layer(19, conv_18, 1024, 3, 1)
        pool_20 = self.pooling_layer(20, conv_19, 2, 2)
        conv_21 = self.conv_layer(21, pool_20, 512, 1, 1)
        conv_22 = self.conv_layer(22, conv_21, 1024, 3, 1)
        conv_23 = self.conv_layer(23, conv_22, 512, 1, 1)
        conv_24 = self.conv_layer(24, conv_23, 1024, 3, 1)
        conv_25 = self.conv_layer(25, conv_24, 1024, 3, 1, trainable=training)
        pad = tf.pad(conv_25, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]))
        conv_26 = self.conv_layer(26, pad, 1024, 3, 2, trainable=training)
        conv_27 = self.conv_layer(27, conv_26, 1024, 3, 1, trainable=training)
        conv_28 = self.conv_layer(28, conv_27, 1024, 3, 1, trainable=training)
        flat = tf.reshape(tf.transpose(conv_28, (0, 3, 1, 2)), (-1, conv_28.shape[1] * conv_28.shape[2] * conv_28.shape[3]))
        fc_29 = self.fc_layer(29, flat, 512, flat=True, linear=False, trainable=training)
        fc_30 = self.fc_layer(30, fc_29, 4096, flat=False, linear=False, trainable=training)
        drop = self.dropout(fc_30, training)
        fc_32 = self.fc_layer(31, drop, self.output_size, flat=False, linear=True, trainable=training)
        return fc_32

    def compute_loss(self, predicts, labels):
        predict_classes = tf.reshape(predicts[:, :self.boundary1],
                                     [-1, self.cell_size, self.cell_size, self.num_class])
        predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2],
                                    [-1, self.cell_size, self.cell_size, self.boxes_per_cell])
        predict_boxes = tf.reshape(predicts[:, self.boundary2:],
                                   [-1, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

        response = tf.reshape(labels[:, :, :, 0], [-1, self.cell_size, self.cell_size, 1])
        boxes = tf.reshape(labels[:, :, :, 1:5], [-1, self.cell_size, self.cell_size, 1, 4])
        boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
        classes = labels[:, :, :, 5:]

        offset = tf.constant(self.offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
        offset = tf.tile(offset, [self.batch_size, 1, 1, 1])

        # x 좌표에 offset을 더한 뒤 cell size로 나눔
        predict_boxes_tran = tf.stack([1. * (predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                       1. * (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (
                                           0, 2, 1, 3))) / self.cell_size,
                                       tf.square(predict_boxes[:, :, :, :, 2]),
                                       tf.square(predict_boxes[:, :, :, :, 3])])
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])
        iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

        # 이거 이상하네
        # 1. IOU 높고 ojbect 있고
        # 2. IOU 높고 ojbect 없고
        # 3. IOU 낮고 ojbect 있고
        # 4. IOU 낮고 ojbect 없고
        # 기존에는 1 번과 2 3 를 사용

        # GT box가 없는 위치에서 confidence를 줄여야 함
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
        # noobject_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * (tf.ones_like(object_mask, dtype=tf.float32) - response)

        boxes_tran = tf.stack([1. * boxes[:, :, :, :, 0] * self.cell_size - offset,
                               1. * boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

        # class_loss
        class_delta = response * (predict_classes - classes)
        self.class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                                         name='class_loss') * self.class_scale

        # object_loss
        # object가 있고 iou를 만족하는 위치에서 1값이 나오게 만들어야함
        # object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_delta = object_mask * (response - predict_scales)
        self.object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                                          name='object_loss') * self.object_scale

        # noobject_loss
        # object가 없는데 iou를 만족하는 위치에서 0값이 나오게 해야함
        noobject_delta = noobject_mask * predict_scales
        self.noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                                            name='noobject_loss') * self.noobject_scale

        # coord_loss
        # iou가 일정 이상 겹치는 애들
        coord_mask = tf.expand_dims(object_mask, 4)
        print('predict_boxes : ', predict_boxes)
        print('boxes_tran : ', boxes_tran)
        print('predict_boxes_tran : ', predict_boxes_tran)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        print('boxes_delta : ', boxes_delta)
        self.coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                                         name='coord_loss') * self.coord_scale

        tf.losses.add_loss(self.class_loss)
        tf.losses.add_loss(self.object_loss)
        tf.losses.add_loss(self.noobject_loss)
        tf.losses.add_loss(self.coord_loss)
        self.total_loss = tf.losses.get_total_loss()

        tf.summary.scalar('class_loss', self.class_loss)
        tf.summary.scalar('object_loss', self.object_loss)
        tf.summary.scalar('noobject_loss', self.noobject_loss)
        tf.summary.scalar('coord_loss', self.coord_loss)
        tf.summary.scalar('total_loss', self.total_loss)

        tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
        tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
        tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
        tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
        tf.summary.histogram('iou', iou_predict_truth)

    def calc_iou(self, boxes1, boxes2):

        boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,  # xmin
                           boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,  # ymin
                           boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,  # xmax
                           boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])  # yamx
        boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

        boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                           boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                           boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
        boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

        lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        intersection = tf.maximum(0.0, rd - lu)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                  (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
        square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                  (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

        union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def train(self, epoch, batch_idx, batch):
        feed_dict = {self.images: batch[0],
                     self.labels: batch[1]}
        loss, c_loss, o_loss, no_loss, coord_loss, _, global_step, summary_str = self.sess.run(
            [self.total_loss, self.class_loss, self.object_loss, self.noobject_loss, self.coord_loss,
             self.optimizer, self.global_step, self.summary_op],
            feed_dict=feed_dict)

        if batch_idx % self.summary_step == 0:
            self.writer.add_summary(summary_str, global_step=global_step)
            self.saver.save(self.sess, os.path.join(self.logs_dir, 'yolo_v1.ckpt'), global_step=global_step)
        return (
            '{} Epoch: {}, Step: {}, total_loss: {:.4f}, class_loss: {:.4f}, object_loss: {:.4f}, nonobj_loss: {:.4f}, coord_loss: {:.4f}').format(
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, batch_idx, loss, c_loss, o_loss,
            no_loss, coord_loss)


    def draw_result(self, img, result):
        colors = self.random_colors(len(result))
        for i in range(len(result)):
            x = int(result[i][1])
            y = int(result[i][2])
            w = int(result[i][3] / 2)
            h = int(result[i][4] / 2)
            color = tuple([rgb * 255 for rgb in colors[i]])
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), color, 1)
            cv2.putText(img, result[i][0], (x - w + 1, y - h + 8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1)
            print(result[i][0], ': %.2f%%' % (result[i][5] * 100))

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        # inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = inputs[:, :, (2, 1, 0)]
        inputs = (inputs / 255.0)
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]
        print(result)

        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result

    def detect_from_cvmat(self, inputs):
        net_output = self.sess.run(self.logits, feed_dict={self.images: inputs})
        print(net_output)
        results = []
        for i in range(net_output.shape[0]):
            print(i)
            results.append(self.interpret_output(net_output[i]))

        return results

    def interpret_output(self, output):
        print(output.shape)
        probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class))
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
        scales = np.reshape(output[self.boundary1:self.boundary2],
                            (self.cell_size, self.cell_size, self.boxes_per_cell))
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))  # 7*7*2
        # print('probs : ', probs.shape)
        # print('class_probs : ', class_probs.shape)
        # print('scales : ', scales.shape)
        # print('boxes : ', boxes.shape)
        # print('offset : ', offset.shape)

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        probs_filtered = probs[filter_mat_probs]
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
                i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

        return result

    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
             max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
             max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        if tb < 0 or lr < 0:
            intersection = 0
        else:
            intersection = tb * lr
        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def random_colors(self, N, bright=True):
        brightness = 1.0 if bright else 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        np.random.shuffle(colors)
        return colors

    def save_cfg(self):
        with open(os.path.join(self.logs_dir, 'config.txt'), 'w', encoding='utf-8') as f:
            dict = config.__dict__
            for key in sorted(dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, dict[key])
                    f.write(cfg_str)
