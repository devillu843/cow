import os

import torch
from torchvision import transforms

from Mydataset import MyDataset2
from read_split_data import read_split_data

# from my_dataset import MyDataSet
# from utils import read_split_data, plot_data_loader_image

# https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
# root = "/home/wz/my_github/data_set/flower_data/flower_photos"  # 数据集所在根目录
root1 = r"E:/硕士/Github搭建工程/dataset/select-14-all-and-head/all"
root2 = r"E:/硕士/Github搭建工程/dataset/select-14-all-and-head/head"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    train_images_path1, train_images_label1, val_images_path1, val_images_label1, val_images_path1, val_images_label1 = read_split_data(
        root1)
    train_images_path2, train_images_label2, val_images_path2, val_images_label2, val_images_path1, val_images_label1 = read_split_data(
        root2)

    # print(train_images_path1, "\n\n\n\n\n\n\n")
    # print(train_images_path2)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    train_data_set = MyDataset2(image_path1=train_images_path1,
                                image_path2=train_images_path2,
                                label=train_images_label1,
                                transform=data_transform["train"])

    batch_size = 8
    # number of workers
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    # print('Using {} dataloader workers'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               collate_fn=train_data_set.collate_fn)

    # plot_data_loader_image(train_loader)

    for step, data in enumerate(train_loader):
        images1, images2, labels = data
        # print(images1)


if __name__ == '__main__':
    main()
