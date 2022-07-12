from random import sample, shuffle

import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    '''
    annotation_lines：需要增强的图片，TODO 格式是什么，第一个是batchsize?
    input_shape: 需要将图片调整到的大小
    num_classes,一共多少类，voc20类，COCO80类
    epoch_length：TODO 和length相乘控制是否进行数据增强，含义还不懂
    mosaic：bool变量，控制是否进行mosaic数据增强，训练时进行增强，验证和测试时不进行增强
    train：模式控制
    mosaic_ratio = 0.9:进行mosaic数据增强的比例

    返回值：[images,box]

    '''

    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, mosaic, train, mosaic_ratio=0.9) -> None:
        super().__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.train = train
        self.mosaic_ratio = mosaic_ratio
        self.step_now = -1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        self.step_now += 1
        #---------------------------------------------------#
        #   在读取数据时就决定是否进行数据增强
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic:
            if self.rand() < 0.5 and self.step_now < self.epoch_length * self.mosaic_ratio * self.length:
                lines = sample(self.annotation_lines, 3)  # sample 随机取三个样本
                lines.append(self.annotation_lines[index])
                shuffle(lines)
                image, box = self.get_random_data_with_Mosaic(
                    lines, self.input_shape)
            else:
                # 此处box返回两点坐标
                image, box = self.get_random_data(
                    self.annotation_lines[index], self.input_shape, random=self.train)
        else:
            image, box = self.get_random_data(
                self.annotation_lines[index], self.input_shape, random=self.train)  # 只能猜测box返回为左上右下坐标
        image = np.transpose(preprocess_input(
            np.array(image, dtype=np.float32)), (2, 0, 1))  # 将通道数移动到最前面
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]  # 得到宽高
            box[:, 0:2] = box[:, 2:4] + box[:, 0:2] / 2  # 得到中心点
        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        '''
        jitter：控制长宽扭曲程度
        hue,sat,val:hsv色域扭曲范围
        random:同类内的train，就是换了个名字
        '''
        line = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih = image.size  # 图像宽高
        h, w = input_shape  # 目标宽高
        #------------------------------#
        #   获得预测框
        #   map(function,x)
        #   对x中的所有数据进行function函数
        #   此处box输出为两点坐标
        #------------------------------#
        box = np.array([np.array(list(map(int, box.split(','))))
                       for box in line[1:]])

        if not random:  # 非训练模式
            scale = min(w/iw, h/ih)  # 选取最小的比值
            nw = int(iw*scale)  # 调整大小
            nh = int(ih*scale)  # 调整大小
            dx = (w-nw)//2  # 需要补上的灰条宽度
            dy = (h-nh)//2  # 需要补上的灰条高度，dx、dy得到了贴图的左上角坐标

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #   box为两点坐标
            #---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx  # nw/iw = scale
                box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy  # nh/ih = scale
                box[:, 0:2][box[:, 2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                # 给的注释是：discard invalid box.丢弃无效的盒子
                box = box[np.logical_and(box_w > 1, box_h > 1)]

            return image_data, box
        #   训练模式
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter, 1+jitter) / \
            self.rand(1-jitter, 1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        #------------------------------------------#
        #   hsv色域扭曲
        #------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1/self.rand(1, val)
        #   这里用的是cv2的cvtColor，不是up自己写的转换
        x = cv2.cvtColor(np.array(image, np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        #---------------------------------#
        #   对真实框进行调整，不懂看前面的注释
        #---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
            box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box

    def merge_bboxes(self, bboxes, cutx, cuty):
        '''
        融合bboxes？？？没懂
        bboxes:框，猜测第一个参数是batch，被len提取出来，mosaic融合四张图！！！！
        cutx:裁剪横坐标，Mosaic的坐标 
        cuty:裁剪纵坐标，Mosaic的坐标
        '''
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                # 一一对应关系，如果不在图片中就什么也没有
                if i == 0:  # cutx,cuty在图片中间
                    if y1 > cuty or x1 > cutx:  # 在图片左上角区域
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:  # cutx,cuty在图片中间
                    if y2 < cuty or x1 > cutx:  # 在图片左下角区域
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:  # cutx,cuty在图片中间
                    if y2 < cuty or x2 < cutx:   # 在图片右下角区域
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:  # cutx,cuty在图片中间
                    if y1 > cuty or x2 < cutx:   # 在图片右上角区域
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, max_boxes=100, hue=.1, sat=1.5, val=1.5):
        '''
        不写了，都是重复的
        '''
        h, w = input_shape
        min_offset_x = self.rand(0.25, 0.75)
        min_offset_y = self.rand(0.25, 0.75)

        # 存了4个随机数
        nws = [int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1)),
               int(w * self.rand(0.4, 1)), int(w * self.rand(0.4, 1))]
        nhs = [int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1)),
               int(h * self.rand(0.4, 1)), int(h * self.rand(0.4, 1))]

        # 对应四张图的分散坐标
        place_x = [int(w*min_offset_x) - nws[0], int(w*min_offset_x) -
                   nws[1], int(w*min_offset_x), int(w*min_offset_x)]
        place_y = [int(h*min_offset_y) - nhs[0], int(h*min_offset_y),
                   int(h*min_offset_y), int(h*min_offset_y) - nhs[3]]

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([np.array(list(map(int, box.split(','))))
                           for box in line_content[1:]])

            # 是否翻转图片
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            nw = nws[index]
            nh = nhs[index]
            image = image.resize((nw, nh), Image.BICUBIC)

            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]]*nw/iw + dx
                box[:, [1, 3]] = box[:, [1, 3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box
            image_datas.append(image_data)
            box_datas.append(box_data)

        #---------------------------------#
        #   先对图像进行偏移，再将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 进行色域变换，这里需要补充HSV域的关系 TODO
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(new_image/255, np.float32),
                         cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        new_image = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        # 对框进行进一步的处理
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes
