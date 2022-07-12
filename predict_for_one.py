import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import os
import time
from sklearn.metrics import classification_report
from sklearn import metrics
from torchsummary import summary
from PIL import Image

from Backbone.AlexNet import AlexNet
from Backbone.ResNet import resnet18, resnet34, resnet101, resnet152, resnet50
from Backbone.ConfusionMatrix import ConfusionMatrix


num_classes = 14
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
json_file = open('class-MIX-144.json', 'r')
class_dict = json.load(json_file)
model = resnet34(num_classes=num_classes).to(device)

labels = [label for _, label in class_dict.items()]
confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)
model.load_state_dict(torch.load('./weights/test-14/ResNet34_train_val.pth'))

image = Image.open(
    './dataset/select-14-all-and-head/head/test-12/1200001104.jpg')

trans = transforms.Compose([transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                0.229, 0.224, 0.225])])
image = trans(image)
image = torch.unsqueeze(image, dim=0)
output = model(image.to(device))
output_pro = F.softmax(output)
print(output_pro)
print(output_pro.sum())

output_pro = output_pro.to('cpu')
predict = torch.max(output_pro)
predict_number = torch.argmax(output_pro).detach().numpy()

print(predict)
print(predict_number)

pred, pred_num = output_pro.topk(k=3)
print(pred)
print(pred_num)
