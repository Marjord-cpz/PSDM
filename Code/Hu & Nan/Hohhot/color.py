import numpy as np
import os
import torch
from tqdm import tqdm
from PIL import Image
from architecture import IHS
from getDataset import getcolorDataset
import time

def verify_second_ms(model, label_loader, unlabel_loader, expo, num, mode='2'):
    if os.path.exists(expo) == 0:
        os.makedirs(expo)
    model.eval()
    label_np1 = np.zeros([size[0], size[1]])
    label_np2 = np.zeros([size[0], size[1]])
    loop = tqdm(label_loader, leave=True)
    with torch.no_grad():
        for ms, pan, target, xy in loop:
            ms, pan, target = ms.cuda(), pan.cuda(), target.cuda()
            out = model(ms)
            pred = out.data.max(1, keepdim=True)[1]
            for i in range(int(ms.shape[0])):
                label_np1[int(xy[i][0])][int(xy[i][1])] = int(pred[i])+1
                label_np2[int(xy[i][0])][int(xy[i][1])] = int(pred[i])+1
            loop.set_postfix(mode='verify')
        loop.close()

    loop = tqdm(unlabel_loader, leave=True)
    with torch.no_grad():
        for ms, pan, xy in loop:
            ms, pan = ms.cuda(), pan.cuda()
            out = model(ms)
            pred = out.data.max(1, keepdim=True)[1]
            for i in range(int(ms.shape[0])):
                label_np2[int(xy[i][0])][int(xy[i][1])] = int(pred[i])+1
            loop.set_postfix(mode='verify')
    label_pic = np.zeros([size[0], size[1], 3])
    for i in range(label_np1.shape[0]):
        for j in range(label_np1.shape[1]):
            label_pic[i][j] = colormap[int(label_np1[i][j])]
    picture1 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath = "Result\Hohhot" + expo + str(num) + "_picms_1.png"
    picture1.save(savepath)

    for i in range(label_np2.shape[0]):
        for j in range(label_np2.shape[1]):
            label_pic[i][j] = colormap[int(label_np2[i][j])]
    picture2 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath = "Result\Hohhot" +expo + str(num) + "_picms_2.png"
    picture2.save(savepath)

def verify_second_pan(model, label_loader, unlabel_loader, expo, num, mode='2'):
    if os.path.exists(expo) == 0:
        os.makedirs(expo)
    model.eval()
    label_np1 = np.zeros([size[0], size[1]])
    label_np2 = np.zeros([size[0], size[1]])
    loop = tqdm(label_loader, leave=True)
    with torch.no_grad():
        for ms, pan, target, xy in loop:
            ms, pan, target = ms.cuda(), pan.cuda(), target.cuda()
            out = model(pan)
            pred = out.data.max(1, keepdim=True)[1]
            for i in range(int(ms.shape[0])):
                label_np1[int(xy[i][0])][int(xy[i][1])] = int(pred[i])+1
                label_np2[int(xy[i][0])][int(xy[i][1])] = int(pred[i])+1
            loop.set_postfix(mode='verify')
        loop.close()

    loop = tqdm(unlabel_loader, leave=True)
    with torch.no_grad():
        for ms, pan, xy in loop:
            ms, pan = ms.cuda(), pan.cuda()
            out = model(pan)
            pred = out.data.max(1, keepdim=True)[1]
            for i in range(int(ms.shape[0])):
                label_np2[int(xy[i][0])][int(xy[i][1])] = int(pred[i])+1
            loop.set_postfix(mode='verify')
    label_pic = np.zeros([size[0], size[1], 3])
    for i in range(label_np1.shape[0]):
        for j in range(label_np1.shape[1]):
            label_pic[i][j] = colormap[int(label_np1[i][j])]
    picture1 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath = "Result\Hohhot" +expo + str(num) + "_pic_1pan.png"
    picture1.save(savepath)

    for i in range(label_np2.shape[0]):
        for j in range(label_np2.shape[1]):
            label_pic[i][j] = colormap[int(label_np2[i][j])]
    picture2 = Image.fromarray(np.uint8(label_pic))
    # picture.show()
    savepath ="Result\Hohhot" + expo + str(num) + "_picpan_2.png"
    picture2.save(savepath)

if __name__=='__main__':
    size=[2001, 2101, 4]
    colormap=[[0, 0, 0], [0, 255, 255], [0, 0, 255], [237, 145, 33],
              [0, 255, 0], [160, 32, 240], [221, 160, 221], [240, 230, 140],
              [255, 0, 0], [255, 255, 0], [0, 255, 127], [255, 0, 255]]
    studentms_path=".\pre trained models\studentms.pkl"
    studentpan_path=".\pre trained models\studentms.pkl"
    studentms=torch.load(studentms_path).cuda()
    studentpan=torch.load(studentpan_path).cuda()
    label_loader,unlabel_loader=getcolorDataset()
    verify_second_ms(studentms,label_loader,unlabel_loader,"image6",1)
    verify_second_ms(studentms,label_loader,unlabel_loader,"image6",1)