# 测试各种函数功能用
from cv2 import imshow
import torch
import pandas as pd
import numpy as np
from PIL import Image,ImageEnhance
import cv2
import random

torch.manual_seed(100)
torch.cuda.manual_seed(100)
torch.cuda.manual_seed_all(100)
pred = torch.rand(8, 14)
# print(pred)
'''
写入excel
'''
# out, pred_num = pred.topk(k=3)
# print(out)
# print(pred_num)

# for i in range(8):
#     for j in range(3):
#         print(pred_num[i][j])


# def pd_toExcel(id1, p1, id2, p2, id3, p3, fileName):  # pandas库储存数据到excel

#     dfData = {  # 用字典设置DataFrame所需数据
#         '第一预测种类': id1,
#         '第一预测可能性': p1,
#         '第二预测种类': id2,
#         '第二预测可能性': p2,
#         '第三预测种类': id3,
#         '第三预测可能性': p3,

#     }
#     df = pd.DataFrame(dfData)  # 创建DataFrame
#     df.to_excel(fileName, index=False)  # 存表，去除原始索引列（0,1,2...）


# fileName = '测试2.xlsx'

# # pd_toExcel(testData, fileName)
# a = range(3)
# print(a)

'''
数据增强测试
'''

img = Image.open("./dataset/test-41-different/all/cow_5/cow_5_8.png")

# 格式转换
# print(type(img))
# img = np.array(img)
# print(type(img))
# img = np.array(img/255, dtype=float)  


# 亮度
# bright_enhancer = ImageEnhance.Brightness(img)
# # 传入调整系数1.2
# bright_img      = bright_enhancer.enhance(0.5)

# contrast_enhancer = ImageEnhance.Contrast(img)
# # 传入调整系数1.2
# contrast_img      = contrast_enhancer.enhance(1.2)

# color_enhancer = ImageEnhance.Color(img)
# # 传入调整系数1.2
# color_img      = color_enhancer.enhance(1.2)
# bright_img.show()
# HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# img = Image.fromarray(img.astype('uint8')).convert('RGB')
# print(type(img))



# 椒盐噪声
# snr = 0.95
# # 把img转化成ndarry的形式
# img_ = np.array(img).copy()
# h, w, c = img_.shape
# # 原始图像的概率（这里为0.9）
# signal_pct = snr
# # 噪声概率共0.1
# noise_pct = (1 - snr)
# # 按一定概率对（h,w,1）的矩阵使用0，1，2这三个数字进行掩码：掩码为0（原始图像）的概率signal_pct，掩码为1（盐噪声）的概率noise_pct/2.，掩码为2（椒噪声）的概率noise_pct/2.
# mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
# # 将mask按列复制c遍
# mask = np.repeat(mask, c, axis=2)
# img_[mask == 1] = 255 # 盐噪声
# img_[mask == 2] = 0  # 椒噪声
# img_ = Image.fromarray(img_.astype('uint8')).convert('RGB') # 转化为PIL的形式
# img_.show()

# percentage=0.2
# img = np.array(img)
# num=int(percentage*img.shape[0]*img.shape[1])#  椒盐噪声点数量
# random.randint(0, img.shape[0])
# img2=img.copy()
# for _ in range(num):
#     X=random.randint(0,img2.shape[0]-1)#从0到图像长度之间的一个随机整数,因为是闭区间所以-1
#     Y=random.randint(0,img2.shape[1]-1)
#     # if random.randint(0,1) ==0: #黑白色概率55开
#     #     img2[X,Y] = (255,255,255)#白色
#     # else:
#     #     img2[X,Y] =(0,0,0)#黑色
#     img2[X,Y] =(0,0,0)#黑色
# img2 = Image.fromarray(img2.astype('uint8')).convert('RGB')
# img2.show()

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# class Solution:
#     def reversePrint(head: ListNode) -> list:
#         reverse = []
#         if head.next==None:
#             reverse.append(head.val)
#             return reverse
#         reverse = Solution.reversePrint(head.next)
#         reverse.append(head.val)
#         return reverse

head = ListNode(1)
head.next = ListNode(2)
head.next.next = ListNode(3)
head.next.next.next = ListNode(4)
head.next.next.next.next = ListNode(5)

# print(Solution.reversePrint(head))


# class Solution:
#     def reverseList(head: ListNode) -> ListNode:
#         cur, pre = head, None
#         while cur:
#             tmp = cur.next # 暂存后继节点 cur.next
#             cur.next = pre # 修改 next 引用指向
#             pre = cur      # pre 暂存 cur
#             cur = tmp      # cur 访问下一节点
#         return pre

# print(Solution.reverseList(head=head))

def reverseLeftWords(s: str, n: int) -> str:
    a = ''
    b = ''
    for i in len(s):
        if i < n:
            a += s(i)
        else:
            b += s(i)
    c = b + a
    return c

print(reverseLeftWords('abcdefghijklmnopqrstuvwxyz',3))