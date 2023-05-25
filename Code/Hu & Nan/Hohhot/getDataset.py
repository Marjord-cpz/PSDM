import numpy as np
import torch
from torch.nn import functional as F
import torch.optim as optim
from libtiff import TIFF
import cv2
from torch.utils.data import Dataset, DataLoader
import os

ms_path='/root/autodl-tmp/data/huhehaote/ms4.tif'
pan_path='/root/autodl-tmp/data/huhehaote/pan.tif'
label_path="/root/autodl-tmp/data/huhehaote/label6.npy"


def getcolorDataset():
    BATCH_SIZE = 64 # 每次喂给的数据量
    Train_Rate = 1   # 将训练集和测试集按比例分开0.01
    Unlabel_Rate=1
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 是否用GPU环视cpu训练

    # 读取图片——ms4
    ms4_tif = TIFF.open(ms_path, mode='r')
    ms4_np = ms4_tif.read_image()
    print('原始ms4图的形状:', np.shape(ms4_np))

    pan_tif = TIFF.open(pan_path, mode='r')
    pan_np = pan_tif.read_image()
    print('原始pan图的形状:', np.shape(pan_np))

    label_np = np.load(label_path)
    print('label数组形状:', np.shape(label_np))
    # print(label_np)

    # ms4与pan图补零  (给图片加边框）
    Ms4_patch_size = 16  # ms4截块的边长
    Interpolation = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REPLICATE： 进行复制的补零操作;
    # cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
    # cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
    # cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdefgh|abcdefgh|abcdefg;

    top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                    int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
    ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的ms4图的形状:', np.shape(ms4_np))

    Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
    top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                    int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
    pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的pan图的形状:', np.shape(pan_np))

    # 按类别比例拆分数据集
    label_np=label_np.astype(np.uint8)
    # label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255

    label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
    print('类标：', label_element)
    print('各类样本数：', element_count)
    Categories_Number = len(label_element) - 1  # 数据的类别数
    print('标注的类别数：', Categories_Number)
    label_row, label_column = np.shape(label_np)  # 获取标签图的行、列
    # print(label_row,label_column)
    '''归一化图片'''
    def to_tensor(image):
        max_i = np.max(image)
        min_i = np.min(image)
        image = (image - min_i) / (max_i - min_i)
        return image

    ground_xy = np.array([[]] * Categories_Number).tolist()   # [[],[],[],[],[],[],[]]  7个类别  生成多个列表的技巧
    ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)  # [800*830, 2] 二维数组
    unlabeled_xy=[]

    #填充数据进入ground数组
    count = 0
    for row in range(label_row):  # 行
        for column in range(label_column):
            ground_xy_allData[count] = [row, column]
            count = count + 1
            if label_np[row][column] != 0:
                ground_xy[int(label_np[row][column])-1].append([row, column])     # 记录属于每个类别的位置集合
            else:
                unlabeled_xy.append([row, column])
    length_unlabel=len(unlabeled_xy)
    using_length=length_unlabel*Unlabel_Rate
    unlabeled_xy=unlabeled_xy[0:int(using_length)]
    print("无标签数据使用了{}组数据".format(len(unlabeled_xy)))            

    # 标签内打乱
    for categories in range(Categories_Number):
        ground_xy[categories] = np.array(ground_xy[categories])
        shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
        np.random.shuffle(shuffle_array)
        ground_xy[categories] = ground_xy[categories][shuffle_array]    #利用np数组的性质进行打乱，但一定要保证都是np类型的数组

    shuffle_array = np.arange(0, label_row * label_column, 1)           
    np.random.shuffle(shuffle_array)
    ground_xy_allData = ground_xy_allData[shuffle_array]
    unlabeled_xy=np.array(unlabeled_xy)

    ground_xy_train = []
    ground_xy_test = []
    label_train = []
    label_test = []

    for categories in range(Categories_Number):
        categories_number = len(ground_xy[categories])
        # print('aaa', categories_number)
        for i in range(categories_number):
            if i < int(categories_number * Train_Rate):
                ground_xy_train.append(ground_xy[categories][i])
            else:
                ground_xy_test.append(ground_xy[categories][i])
        label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
        label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]
    # length_unlabel=len(unlabeled_xy)
    # using_length=length_unlabel*Unlabel_Rate
    # unlabeled_xy=unlabeled_xy[0:int(using_length)]
    
    label_train = np.array(label_train)
    label_test = np.array(label_test)
    ground_xy_train = np.array(ground_xy_train)
    ground_xy_test = np.array(ground_xy_test)


    # 训练数据与测试数据，数据集内打乱
    shuffle_array = np.arange(0, len(label_test), 1)        #此类打乱操作可以保证打乱的顺序一致，保证数据和标签对应起来
    np.random.shuffle(shuffle_array)
    label_test = label_test[shuffle_array]
    ground_xy_test = ground_xy_test[shuffle_array]

    shuffle_array = np.arange(0, len(label_train), 1)
    np.random.shuffle(shuffle_array)
    label_train = label_train[shuffle_array]
    ground_xy_train = ground_xy_train[shuffle_array]


    label_train = torch.from_numpy(label_train).type(torch.LongTensor)
    label_test = torch.from_numpy(label_test).type(torch.LongTensor)
    ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
    ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
    ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)
    unlabeled_xy=torch.from_numpy(unlabeled_xy).type(torch.LongTensor)



    # 数据归一化
    ms4 = to_tensor(ms4_np)
    pan = to_tensor(pan_np)
    pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
    ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道

    # 转换类型
    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)
    
    class MyData(Dataset):
        def __init__(self, MS4, Pan, Label, xy, cut_size):
            self.train_data1 = MS4
            self.train_data2 = Pan
            self.train_labels = Label
            self.gt_xy = xy
            self.cut_ms_size = cut_size
            self.cut_pan_size = cut_size * 4

        def __getitem__(self, index):
            x_ms, y_ms = self.gt_xy[index]
            x_pan = int(4 * x_ms)      # 计算不可以在切片过程中进行
            y_pan = int(4 * y_ms)
            image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                    y_ms:y_ms + self.cut_ms_size]

            image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                        y_pan:y_pan + self.cut_pan_size]

            locate_xy = self.gt_xy[index]

            target = self.train_labels[index]
            return image_ms, image_pan, target, locate_xy

        def __len__(self):
            return len(self.gt_xy)

    class MyData1(Dataset):
        def __init__(self, MS4, Pan, xy, cut_size):
            self.train_data1 = MS4
            self.train_data2 = Pan

            self.gt_xy = xy
            self.cut_ms_size = cut_size
            self.cut_pan_size = cut_size * 4

        def __getitem__(self, index):
            x_ms, y_ms = self.gt_xy[index]
            x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
            y_pan = int(4 * y_ms)
            image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                    y_ms:y_ms + self.cut_ms_size]

            image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                        y_pan:y_pan + self.cut_pan_size]

            locate_xy = self.gt_xy[index]

            return image_ms, image_pan, locate_xy

        def __len__(self):
            return len(self.gt_xy)

    train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
    unlabeled_data=MyData1(ms4,pan,unlabeled_xy,Ms4_patch_size)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,drop_last=True)
    unlabeled_loader=DataLoader(dataset=unlabeled_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,drop_last=True)

    print("unlabel length is {}".format(len(unlabeled_loader.dataset)))
    print("label length is {}".format(len(train_loader.dataset)))
    return train_loader,unlabeled_loader













def getDataset():
    BATCH_SIZE = 64 # 每次喂给的数据量
    Train_Rate = 0.01   # 将训练集和测试集按比例分开0.01
    Unlabel_Rate=0.01
    Test_rate=0.05
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'   # 是否用GPU环视cpu训练

    # 读取图片——ms4
    ms4_tif = TIFF.open(ms_path, mode='r')
    ms4_np = ms4_tif.read_image()
    print('原始ms4图的形状:', np.shape(ms4_np))

    pan_tif = TIFF.open(pan_path, mode='r')
    pan_np = pan_tif.read_image()
    print('原始pan图的形状:', np.shape(pan_np))

    label_np = np.load(label_path)
    print('label数组形状:', np.shape(label_np))

    # ms4与pan图补零  (给图片加边框）
    Ms4_patch_size = 16  # ms4截块的边长
    Interpolation = cv2.BORDER_REFLECT_101
    # cv2.BORDER_REPLICATE： 进行复制的补零操作;
    # cv2.BORDER_REFLECT:  进行翻转的补零操作:gfedcba|abcdefgh|hgfedcb;
    # cv2.BORDER_REFLECT_101： 进行翻转的补零操作:gfedcb|abcdefgh|gfedcb;
    # cv2.BORDER_WRAP: 进行上下边缘调换的外包复制操作:bcdefgh|abcdefgh|abcdefg;

    top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                    int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
    ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的ms4图的形状:', np.shape(ms4_np))

    Pan_patch_size = Ms4_patch_size * 4  # pan截块的边长
    top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                    int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
    pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
    print('补零后的pan图的形状:', np.shape(pan_np))

    # 按类别比例拆分数据集
    label_np=label_np.astype(np.uint8)
    label_np = label_np - 1  # 标签中0类标签是未标注的像素，通过减一后将类别归到0-N，而未标注类标签变为255

    label_element, element_count = np.unique(label_np, return_counts=True)  # 返回类别标签与各个类别所占的数量
    print('类标：', label_element)
    print('各类样本数：', element_count)
    Categories_Number = len(label_element) - 1  # 数据的类别数
    print('标注的类别数：', Categories_Number)
    label_row, label_column = np.shape(label_np)  # 获取标签图的行、列
    # print(label_row,label_column)
    '''归一化图片'''
    def to_tensor(image):
        max_i = np.max(image)
        min_i = np.min(image)
        image = (image - min_i) / (max_i - min_i)
        return image

    ground_xy = np.array([[]] * Categories_Number).tolist()   # [[],[],[],[],[],[],[]]  7个类别  生成多个列表的技巧
    ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)  # [800*830, 2] 二维数组
    unlabeled_xy=[]

    #填充数据进入ground数组
    count = 0
    for row in range(label_row):  # 行
        for column in range(label_column):
            ground_xy_allData[count] = [row, column]
            count = count + 1
            if label_np[row][column] != 255:
                ground_xy[int(label_np[row][column])].append([row, column])     # 记录属于每个类别的位置集合
            else:
                unlabeled_xy.append([row, column])
    length_unlabel=len(unlabeled_xy)
    using_length=length_unlabel*Unlabel_Rate
    unlabeled_xy=unlabeled_xy[0:int(using_length)]
    print("无标签数据使用了{}组数据".format(len(unlabeled_xy)))            

    # 标签内打乱
    for categories in range(Categories_Number):
        ground_xy[categories] = np.array(ground_xy[categories])
        shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
        np.random.shuffle(shuffle_array)
        ground_xy[categories] = ground_xy[categories][shuffle_array]    #利用np数组的性质进行打乱，但一定要保证都是np类型的数组

    shuffle_array = np.arange(0, label_row * label_column, 1)           
    np.random.shuffle(shuffle_array)
    ground_xy_allData = ground_xy_allData[shuffle_array]
    unlabeled_xy=np.array(unlabeled_xy)

    ground_xy_train = []
    ground_xy_test = []
    label_train = []
    label_test = []

    for categories in range(Categories_Number):
        categories_number = len(ground_xy[categories])
        # print('aaa', categories_number)
        for i in range(categories_number):
            if i < int(categories_number * Train_Rate):
                ground_xy_train.append(ground_xy[categories][i])
            else:
                ground_xy_test.append(ground_xy[categories][i])
        label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
        label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]
    # length_unlabel=len(unlabeled_xy)
    # using_length=length_unlabel*Unlabel_Rate
    # unlabeled_xy=unlabeled_xy[0:int(using_length)]
    
    label_train = np.array(label_train)
    label_test = np.array(label_test)
    ground_xy_train = np.array(ground_xy_train)
    ground_xy_test = np.array(ground_xy_test)


    # 训练数据与测试数据，数据集内打乱
    shuffle_array = np.arange(0, len(label_test), 1)        #此类打乱操作可以保证打乱的顺序一致，保证数据和标签对应起来
    np.random.shuffle(shuffle_array)
    label_test = label_test[shuffle_array]
    ground_xy_test = ground_xy_test[shuffle_array]

    shuffle_array = np.arange(0, len(label_train), 1)
    np.random.shuffle(shuffle_array)
    label_train = label_train[shuffle_array]
    ground_xy_train = ground_xy_train[shuffle_array]

    Test_length=int(Test_rate*len(label_test))
    Test_label=label_test[0:Test_length]
    Test_xy=ground_xy_test[0:Test_length]

    Test_label=torch.from_numpy(Test_label).type(torch.LongTensor)
    Test_xy=torch.from_numpy(Test_xy).type(torch.LongTensor)
    label_train = torch.from_numpy(label_train).type(torch.LongTensor)
    label_test = torch.from_numpy(label_test).type(torch.LongTensor)
    ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
    ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
    ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)
    unlabeled_xy=torch.from_numpy(unlabeled_xy).type(torch.LongTensor)



    # 数据归一化
    ms4 = to_tensor(ms4_np)
    pan = to_tensor(pan_np)
    pan = np.expand_dims(pan, axis=0)  # 二维数据进网络前要加一维
    ms4 = np.array(ms4).transpose((2, 0, 1))  # 调整通道

    # 转换类型
    ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
    pan = torch.from_numpy(pan).type(torch.FloatTensor)
    
    class MyData(Dataset):
        def __init__(self, MS4, Pan, Label, xy, cut_size):
            self.train_data1 = MS4
            self.train_data2 = Pan
            self.train_labels = Label
            self.gt_xy = xy
            self.cut_ms_size = cut_size
            self.cut_pan_size = cut_size * 4

        def __getitem__(self, index):
            x_ms, y_ms = self.gt_xy[index]
            x_pan = int(4 * x_ms)      # 计算不可以在切片过程中进行
            y_pan = int(4 * y_ms)
            image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                    y_ms:y_ms + self.cut_ms_size]

            image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                        y_pan:y_pan + self.cut_pan_size]

            locate_xy = self.gt_xy[index]

            target = self.train_labels[index]
            return image_ms, image_pan, target, locate_xy

        def __len__(self):
            return len(self.gt_xy)

    class MyData1(Dataset):
        def __init__(self, MS4, Pan, xy, cut_size):
            self.train_data1 = MS4
            self.train_data2 = Pan

            self.gt_xy = xy
            self.cut_ms_size = cut_size
            self.cut_pan_size = cut_size * 4

        def __getitem__(self, index):
            x_ms, y_ms = self.gt_xy[index]
            x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
            y_pan = int(4 * y_ms)
            image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                    y_ms:y_ms + self.cut_ms_size]

            image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                        y_pan:y_pan + self.cut_pan_size]

            locate_xy = self.gt_xy[index]

            return image_ms, image_pan, locate_xy

        def __len__(self):
            return len(self.gt_xy)

    train_data = MyData(ms4, pan, label_train, ground_xy_train, Ms4_patch_size)
    test_data = MyData(ms4, pan, label_test, ground_xy_test, Ms4_patch_size)
    all_data = MyData1(ms4, pan, ground_xy_allData, Ms4_patch_size)
    Test_data=MyData(ms4, pan, Test_label, Test_xy, Ms4_patch_size)
    unlabeled_data=MyData1(ms4,pan,unlabeled_xy,Ms4_patch_size)
    Test_loader = DataLoader(dataset=Test_data, batch_size=BATCH_SIZE,shuffle=False,num_workers=0,drop_last=True)
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,drop_last=True)

    unlabeled_loader=DataLoader(dataset=unlabeled_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0,drop_last=True)
    print("test length is {}".format(len(Test_loader.dataset)))
    print("label length is {}".format(len(train_loader.dataset)))
    return train_loader,Test_loader,unlabeled_loader










def GetSpecialDataset(ms,pan,label):

    class Filter_Data(Dataset):
        def __init__(self,ms,pan,label):
            self.size=ms.size()[0]
            self.ms_patch=ms
            self.pan_patch=pan
            self.label=label
        def __getitem__(self,index):
            ms_patch=self.ms_patch[index]
            pan_patch=self.pan_patch[index]
            label=self.label[index]
            return ms_patch,pan_patch,label
        def __len__(self):
            return len(self.label)
    batch_size=64
    filter_data=Filter_Data(ms,pan,label)
    filter_loader=DataLoader(filter_data,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=True)
    return filter_loader
    pass
