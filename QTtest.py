import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from Ui_demo_test import Ui_MainWindow

from PyQt5 import QtGui,QtCore
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt,QRect


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, datasets

from tqdm import tqdm
from PIL import Image
import json

from myutils.Mydataset import MyDataset, MyDataset_QT
from myutils.write_into_file import pd_toExcel


from Backbone.AlexNet import AlexNet
from Backbone.ResNet import resnet18, resnet34, resnet101, resnet152, resnet50



class MyClass(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyClass, self).__init__(parent)
        self.setupUi(self)
        self.imgPath = []
        self.pushButton.clicked.connect(self.open_one_image)
        # self.pushButton_2.clicked.connect(self.open_images)
        self.pushButton_3.clicked.connect(self.classes_for_one_image)
        # self.pushButton_4.clicked.connect(self.classes_for_images)
   
    def open_one_image(self):
        self.imgName, imgType = QFileDialog.getOpenFileName(
            self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        jpg = QtGui.QPixmap(self.imgName)
        if jpg.width()>880 or jpg.height()>1320:
            scale = min(jpg.width()/880,jpg.height()/1320)
            
            jpg = jpg.scaled(jpg.width()*scale, jpg.height()*scale)
        self.label_5.setPixmap(jpg)
        self.label_5.setAlignment(Qt.AlignCenter)



    # def open_images(self):
    #     self.imgPath, imgType = QFileDialog.getOpenFileNames(
    #         self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
    #     for i in self.imgPath:
    #         text = self.label_5.text() + '\n' + i
    #         self.label_5.setText(text)
    #     self.label_5.setAlignment(Qt.AlignTop)
    
    def classes_for_one_image(self):
        # excel_name = self.lineEdit_3.text()
        json_file = open('class-test-14.json', 'r')
        class_indict = json.load(json_file)
        batch_size = 8
        data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                        0.229, 0.224, 0.225])
                                    ])
        
        img = Image.open(self.imgName)        
        img = data_transform(img)
        img = torch.unsqueeze(img, dim=0)
        
        # ------------------------------------------
        # 模型调整
        # ------------------------------------------
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(device)

        model = AlexNet(num_classes=14).to(device)
        model.load_state_dict(torch.load('./weights/test-14/Alex_train_val.pth'))
        model.eval()

        output = torch.squeeze(model(img.to(device)))
        output = F.softmax(output, dim=0)
        k = 3
        out, pred_num = output.topk(k=k)
        cla = ''
        

        
        for j in range(k):
            # print(pred_num[i][j])
            cow_name = class_indict[str(
                pred_num[j].to('cpu').numpy())]
            probability = out[j].to('cpu').detach().numpy() * 100
            if j == 0:
                cla = cla +'识别结果为:'+str(cow_name)+'，正确率为：'+str(probability)[0:5]+'%'
            # elif j == 1:
            #     cla = cla +'识别结果2为:'+str(cow_name)+'，可能性为：'+str(probability)+'\n'
            # else:
            #     cla = cla +'识别结果3为:'+str(cow_name)+'，可能性为：'+str(probability)+'\n'
        self.listWidget_2.addItem(cla)

        
        


    # def classes_for_images(self):
    #     excel_name = self.lineEdit_3.text()
    #     json_file = open('class-test-14.json', 'r')
    #     class_indict = json.load(json_file)
    #     batch_size = 8
    #     data_transform = {
    #         'test': transforms.Compose([transforms.Resize((224, 224)),
    #                                     transforms.ToTensor(),
    #                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
    #                                         0.229, 0.224, 0.225])
    #                             ])        
    #                         }
        
    #     test_data_set = MyDataset_QT(image_path=self.imgPath,
    #                       transform=data_transform["test"])
    #     test_num = len(test_data_set)
    #     test_loader = torch.utils.data.DataLoader(test_data_set,
    #                                       batch_size=batch_size,
    #                                       shuffle=False,
    #                                       num_workers=0,
    #                                       drop_last=False,
    #                                     #   collate_fn=test_data_set.collate_fn
    #                                       )
    #     # ------------------------------------------
    #     # 模型调整
    #     # ------------------------------------------
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     print(device)

    #     model = AlexNet(num_classes=14).to(device)
    #     model.load_state_dict(torch.load('./weights/test-14/Alex_train_val.pth'))
    #     model.eval()
    #     with torch.no_grad():
    #         loop_test = tqdm(enumerate(test_loader), total=len(test_loader))
    #         id1 = []
    #         p1 = []
    #         id2 = []
    #         p2 = []
    #         id3 = []
    #         p3 = []
    #         batch_num = 1

    #         for step,test_images in loop_test:
    #             # print(test_images.shape)

    #             outputs = model(test_images.to(device))  # 指认设备

    #             # -----------------------------------------------
    #             # 保存k个数据，与训练本身无关，郑老师的要求,写入excel文件
    #             k = 3
    #             output = F.softmax(outputs)
    #             out, pred_num = output.topk(k=k)

    #             if batch_num*batch_size <= test_num:
    #                 image_num_in_batch = batch_size
    #                 batch_num += 1
    #             else:
    #                 image_num_in_batch = batch_size - batch_size*batch_num + test_num

    #             for i in range(image_num_in_batch):
    #                 for j in range(k):
    #                     # print(pred_num[i][j])
    #                     cow_name = class_indict[str(
    #                         pred_num[i][j].to('cpu').numpy())]
    #                     probability = out[i][j].to('cpu').numpy() * 100
    #                     if j == 0:
    #                         id1.append(cow_name)
    #                         p1.append(probability)
    #                     elif j == 1:
    #                         id2.append(cow_name)
    #                         p2.append(probability)
    #                     else:
    #                         id3.append(cow_name)
    #                         p3.append(probability)
    #             # -----------------------------------------------
    #     pd_toExcel(self.imgPath, id1, p1, id2, p2, id3, p3, excel_name)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('./qt/images/logo.jpg'))
    myWin = MyClass()
    myWin.setWindowTitle('保险身份识别')

    myWin.show()
    sys.exit(app.exec_())
