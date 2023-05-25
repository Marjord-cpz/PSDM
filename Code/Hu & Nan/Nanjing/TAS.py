import torch
import torch.nn.functional as F
import architecture
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from getDataset import getDataset
import numpy as np
import random
import math

def KLP(teacher_model,training_data,th):
    tot_length=len(training_data.dataset)
    filter_data=[[],[],[]]
    difficult_data=[[],[],[]]
    index_data=[]
    teacher_model.eval()
    with torch.no_grad():
        for ms,pan,label,_ in tqdm(training_data):
            ms,pan,label=ms.cuda(),pan.cuda(),label.cuda()
            fusion_data=architecture.IHS(ms,pan)
            output_teacher=teacher_model(fusion_data)
            output_teacher=F.softmax(output_teacher,dim=1)
            # print(output_teacher.size())
            finds=torch.max(output_teacher,dim=1)
            score=finds[0]
            # print(score[1])
            pred_teacher=output_teacher.max(1,keepdim=True)[1]
            # print(pred_teacher.size())
            pred_teacher=finds[1]  # torch b*1   对应的预测概率的最大类
            # print(pred_teacher.size())
            correct_vector=pred_teacher.eq(label.view_as(pred_teacher).long()) #b*1     预测正确的向量
            if correct_vector.sum()!=output_teacher.size()[0]:
                store_index=torch.nonzero(correct_vector==1)[:,0]
                error_index=torch.nonzero(correct_vector==0)[:,0]
                difficult_data[0].append(ms[error_index])
                difficult_data[1].append(pan[error_index])
                difficult_data[2].append(label[error_index])
            else:
                store_index=torch.nonzero(pred_teacher>=0)[:,0]
            filter_data[0].append(ms[store_index])
            filter_data[1].append(pan[store_index])
            filter_data[2].append(label[store_index])
            index_data.append(score[store_index])
        
        sortedinfo=torch.sort(torch.cat(index_data,dim=0))
        filter_length=int(tot_length*th)
        l_filter=[x for x in range(filter_length)]
        filter_data[0]=torch.cat(filter_data[0],dim=0)[sortedinfo[1],:,:,:][l_filter,:,:,:]
        # print(filter_data[0].size())
        filter_data[1]=torch.cat(filter_data[1],dim=0)[sortedinfo[1],:,:,:][l_filter,:,:,:]
        # print(filter_data[1].size())
        filter_data[2]=torch.cat(filter_data[2],dim=0)[sortedinfo[1]][l_filter]
        # print(filter_data[2].size())
        if difficult_data[0]!=[]:
            difficult_data[0]=torch.cat(difficult_data[0],dim=0)
            difficult_data[1]=torch.cat(difficult_data[1],dim=0)
            difficult_data[2]=torch.cat(difficult_data[2],dim=0)
        # print(difficult_data[0].size())
        # print(difficult_data[1].size())
        # print(difficult_data[2].size())
    return filter_data[0],filter_data[1],filter_data[2],difficult_data[0],difficult_data[1],difficult_data[2]
    pass


def updata_th(th,epo,EPO,S):
    dth=(epo/(EPO**2))*maxf(0,(1-math.exp(-S)))
    flag=0
    if dth==0:
        flag=1
    th+=dth
    print("threshold is {:.4f}, flag is {}".format(th,flag))
    return th,flag
    pass

def updata_TAE(DI,epo,EPO,BaseLine,S_t,TAE):
    dTAE=(1/DI)*(epo/(EPO**2))*maxf(0,(BaseLine-S_t))    
    TAE+=dTAE
    return TAE

def Test_Sudents(student1_model,student2_model,test_loader,TAE):
    student1_model.eval()
    student2_model.eval()
    length=len(test_loader.dataset)
    correct1=0.0
    correct2=0.0
    loss1=0.0
    loss2=0.0
    with torch.no_grad():
        for ms,pan,label,_ in tqdm(test_loader):
            ms,pan,label=ms.cuda(),pan.cuda(),label.cuda()
            output_student1=student1_model(pan)
            output_student2=student2_model(ms)
            loss1+=F.cross_entropy(output_student1,label.long())
            loss2+=F.cross_entropy(output_student2,label.long())
            pred_student1=output_student1.max(1,keepdim=True)[1]
            pred_student2=output_student2.max(1,keepdim=True)[1]
            correct1+=pred_student1.eq(label.view_as(pred_student1).long()).sum().item()
            correct2+=pred_student2.eq(label.view_as(pred_student2).long()).sum().item()
    test_acc1=correct1*100/length
    test_acc2=correct2*100/length
    test_loss1=loss1/length
    test_loss2=loss2/length
    print(" Student1's acc is {:.2f} %, loss is {:.4f}\n Student2's acc is {:.2f} %, loss is {:.4f}.\n".format(test_acc1,test_loss1,test_acc2,test_loss2))
    # S=(1/2)*((test_acc1-TAE)+(test_acc2-TAE))
    return test_acc1/100,test_acc2/100

def GenS(Acc_1,Acc_2,TAE):
    S=(1/2)*((Acc_1-TAE)+(Acc_2-TAE))
    return S

def GenDI(th,type):
    if type==0:
        DI=th+1
    else:
        DI=2
    return DI

def maxf(a,b):
    if a>b:
        return a
    else:
        return b

def Test_Teacher(teacher,test_loader):
    teacher.eval()
    correct=0.0
    loss=0.0
    with torch.no_grad():
        for ms,pan,label,_ in tqdm(test_loader):
            ms,pan,label=ms.cuda(),pan.cuda(),label.cuda()
            fusion_data=architecture.IHS(ms,pan).cuda()
            teacher_output=teacher(fusion_data)
            loss+=F.cross_entropy(teacher_output,label.long())
            pred_teacher=teacher_output.max(1,keepdim=True)[1]
            correct+=pred_teacher.eq(label.view_as(pred_teacher).long()).sum().item()
    Test_Acc=correct*100/len(test_loader.dataset)
    Test_Loss=loss/len(test_loader.dataset)
    print("Test Acc: {:.4f}%\t Test Loss: {:.6f}".format(Test_Acc,Test_Loss))
    return Test_Acc/100

if __name__=='__main__':
# hyparameters
#主要任务：过滤出来数据，给G-SN学习

    # loading model
    teacher=torch.load(".\pre trained models\teacher.pkl").cuda()
    # Dataset
    train_loader,test_loader,unlabeled_loader=getDataset()

    KLP(teacher_model=teacher,training_data=train_loader,th=0.1)