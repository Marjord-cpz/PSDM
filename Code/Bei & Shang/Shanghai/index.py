import numpy as np
from tqdm import tqdm
from getDataset import getDataset
import torch

colormap=10

def test_second(model_pan, model_ms, test_loader, mode='2'):
    model_ms.eval()
    model_pan.eval()
    test_matrix_pan = np.zeros([(colormap), (colormap)])
    test_matrix_ms = np.zeros([(colormap), (colormap)])
    loop = tqdm(test_loader, leave=True)
    with torch.no_grad():
        for data1, data2, target,_ in loop:
            data1, data2, target = data1.cuda(), data2.cuda(), target.cuda()
            output_ms = model_ms(data1)
            output_pan = model_pan(data2)
            pred_ms = output_ms.data.max(1, keepdim=True)[1]
            pred_pan = output_pan.data.max(1, keepdim=True)[1]
            for i in range(len(target)):
                test_matrix_pan[int(pred_pan[i].item())][int(target[i].item())] += 1
                test_matrix_ms[int(pred_ms[i].item())][int(target[i].item())] += 1
            loop.set_postfix(mode='test')
        loop.close()
    return test_matrix_pan,test_matrix_ms

def aa_oa(matrix):
    accuracy = []
    b = np.sum(matrix, axis=0)
    c = 0
    on_display = []
    for i in range(1, matrix.shape[0]+1):
        a = matrix[i-1][i-1]/b[i-1]
        c += matrix[i-1][i-1]
        accuracy.append(a)
        on_display.append([b[i-1], matrix[i-1][i-1], a])
        print("Category:{}. Overall:{}. Correct:{}. Accuracy:{:.6f}".format(i, b[i-1], matrix[i-1][i-1], a))
    aa = np.mean(accuracy)
    oa = c / np.sum(b, axis=0)
    k = kappa(matrix)
    print("OA:{:.6f} AA:{:.6f} Kappa:{:.6f}".format(oa, aa, k))
    return [aa, oa, k, on_display]

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)

if __name__=='__main__':
    Student1_path='/root/autodl-tmp/Shanghai/Version2/pre trained models/studentpan.pkl'
    Student2_path='/root/autodl-tmp/Shanghai/Version2/pre trained models/studentms.pkl'
    Student1=torch.load(Student1_path).cuda()
    Student2=torch.load(Student2_path).cuda()
    train_loader,test_loader,unlabeled_loader=getDataset()
    matrix_pan,matrix_ms=test_second(Student1,Student2,test_loader)
    # print(matrix_pan)
    # print(matrix_ms)
    print("OA,AA,Kappa as follow")
    aa_oa(matrix_pan)
    aa_oa(matrix_ms)
