import architecture
from tqdm import tqdm
from getDataset import getDataset,GetSpecialDataset
from index import test_second,aa_oa
import TAS
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os


def labeled_train(student1_model,student2_model,teacher_model,train_loader,optimizer1,optimizer2,DI,epo,EPO,punish=0):
    student1_model.train()
    student2_model.train()
    teacher_model.eval()
    correct_1=0.0
    correct_2=0.0
    temp=3
    alpha=0.7
    if punish:
        print("Focus on the error!")
        punish=0
    for ms,pan,label,_ in tqdm(train_loader):
        ms,pan,label=ms.cuda(),pan.cuda(),label.cuda()
        fusion_data=architecture.IHS(ms,pan)
        with torch.no_grad():
            output_teacher=teacher_model(fusion_data)
            # pred_teacher=output_teacher.max(1,keepdim=True)[1]
        output_student1=student1_model(pan)
        output_student2=student2_model(ms)
        pred_student1=output_student1.max(1,keepdim=True)[1]
        pred_student2=output_student2.max(1,keepdim=True)[1]
        if punish:
            pf=1+(epo/EPO)*(1/DI)
            correct_vector1=pred_student1.eq(label.view_as(pred_student1).long())   #b*1
            error_index1=torch.nonzero(correct_vector1==0)[:,0]
            right_index1=torch.nonzero(correct_vector1==1)[:,0]
            correct_vector2=pred_student1.eq(label.view_as(pred_student1).long())   #b*1
            error_index2=torch.nonzero(correct_vector2==0)[:,0]
            right_index2=torch.nonzero(correct_vector2==1)[:,0]
            correct_1+=pred_student1.eq(label.view_as(pred_student1).long()).sum().item()
            correct_2+=pred_student2.eq(label.view_as(pred_student2).long()).sum().item()
            eloss_t1=F.kl_div(F.log_softmax(output_student1[error_index1,:]/temp,dim=1),
                             F.softmax(output_teacher[error_index1,:]/temp,dim=1),reduction='batchmean')
            eloss_t2=F.kl_div(F.log_softmax(output_student2[error_index2,:]/temp,dim=1),
                             F.softmax(output_teacher[error_index2,:]/temp,dim=1),reduction='batchmean')
            eloss_ce1=F.cross_entropy(output_student1[error_index1,:],label[error_index1].long())
            eloss_ce2=F.cross_entropy(output_student2[error_index2,:],label[error_index2].long())

            rloss_t1=F.kl_div(F.log_softmax(output_student1[right_index1,:]/temp,dim=1),
                             F.softmax(output_teacher[right_index1,:]/temp,dim=1),reduction='batchmean')
            rloss_t2=F.kl_div(F.log_softmax(output_student2[right_index2,:]/temp,dim=1),
                             F.softmax(output_teacher[right_index2,:]/temp,dim=1),reduction='batchmean')
            rloss_ce1=F.cross_entropy(output_student1[right_index1,:],label[right_index1].long())
            rloss_ce2=F.cross_entropy(output_student2[right_index2,:],label[right_index2].long())
            L_e=eloss_ce1+eloss_ce2+eloss_t1+eloss_t2
            L_r=rloss_ce1+rloss_ce2+rloss_t1+rloss_t2
            loss=L_r+pf*L_e
        else:
            
            loss_t1=F.kl_div(F.log_softmax(output_student1/temp,dim=1),F.softmax(output_teacher/temp,dim=1),reduction='batchmean')
            loss_t2=F.kl_div(F.log_softmax(output_student2/temp,dim=1),F.softmax(output_teacher/temp,dim=1),reduction='batchmean')
            loss_ce1=F.cross_entropy(output_student1,label.long())
            loss_ce2=F.cross_entropy(output_student2,label.long())
            correct_1+=pred_student1.eq(label.view_as(pred_student1).long()).sum().item()
            correct_2+=pred_student2.eq(label.view_as(pred_student2).long()).sum().item()
            loss_student1= (1-alpha)*loss_t1+alpha*loss_ce1
            loss_student2= (1-alpha)*loss_t2+alpha*loss_ce2
            loss=loss_student1+loss_student2

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        loss.backward()
        optimizer1.step()
        optimizer2.step()
    train_acc_pan=correct_1*100/len(train_loader.dataset)
    train_acc_ms=correct_2*100/len(train_loader.dataset)
    print("Train Acc of pan: {:.4f}% \nTrain Acc of ms: {:.4f}% \nThe current loss is {:.6f}".format(train_acc_pan,train_acc_ms,loss))
    pass

def unlabeled_train(student1_model,student2_model,teacher_model,unlabeled_loader,optimizer1,optimizer2):
    # s1: pan
    student1_model.train()
    # s2: ms
    student2_model.train()
    # t: fusion_data
    teacher_model.eval()
    length=len(unlabeled_loader)
    shift_length=int(length*0.5)
    gap_length=int(length*0.1)
    stage_length=[shift_length+gap_length*i for i in range(5)]
    stage_length.append(length)
    stage=-1
    # print(length)
    temp=3
    weight=0.3
    for step,(ms,pan,_) in enumerate(tqdm(unlabeled_loader)):
        ms,pan=ms.cuda(),pan.cuda()
        fusion_data=architecture.IHS(ms,pan)
        with torch.no_grad():
            output_teacher=teacher_model(fusion_data)
            pred_teacher=output_teacher.max(1,keepdim=True)[1]
        output_student1=student1_model(pan)
        output_student2=student2_model(ms)
        loss12=F.kl_div(F.log_softmax(output_student1/temp,dim=1),F.softmax(output_student2/temp,dim=1),reduction='batchmean')
        loss21=F.kl_div(F.log_softmax(output_student2/temp,dim=1),F.softmax(output_student1/temp,dim=1),reduction='batchmean')
        loss_t1=F.cross_entropy(output_student1,pred_teacher.reshape(len(pred_teacher)))
        loss_t2=F.cross_entropy(output_student2,pred_teacher.reshape(len(pred_teacher)))
        # loss_t1=F.cross_entropy(output_student1,pred_teacher)
        # loss_t2=F.cross_entropy(output_student2,pred_teacher)
        if (step-shift_length)%gap_length==0 and stage<len(stage_length)-1:
            stage+=1
        if stage_length[stage]<step<=stage_length[stage+1]==0:
            loss12,loss21,loss_t1,loss_t2=competitive_learning(loss12,loss21,loss_t1,loss_t2)

        loss1=weight*loss12+(1-weight)*loss_t1
        loss2=weight*loss21+(1-weight)*loss_t2
        total_loss=loss1+loss2

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        total_loss.backward()
        optimizer1.step()
        optimizer2.step()
    print(" The student1's current loss is {:.4f}\n The student2's current loss is {:.4f}".format(loss1,loss2))

    pass

def competitive_learning(loss12,loss21,losst1,losst2):
    if losst1<losst2:
        loss12=loss21
    if losst2<losst1:
        loss21=loss12
    return loss12,loss21,losst1,losst2


def Testing_GSN(student1_model,student2_model,test_loader):
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
    return test_acc1/100,test_acc2/100,test_loss1,test_loss2
    pass

def Testing_Teacher(teacher_model,test_loader):
    teacher_model.eval()
    length=len(test_loader.dataset)
    correct=0.0
    loss=0.0
    with torch.no_grad():
        for ms,pan,label in tqdm(test_loader):
            ms,pan,label=ms.cuda(),pan.cuda(),label.cuda()
            fusion_data=architecture.IHS(ms,pan)
            output_teacher=teacher_model(fusion_data)
            loss+=F.cross_entropy(output_teacher,label.long())
            pred_teacher=output_teacher.max(1,keepdim=True)[1]
            correct+=pred_teacher.eq(label.view_as(pred_teacher).long()).sum().item()
    test_acc=correct*100/length
    test_loss=loss/length
    print("Teacher's acc is {:.2f}%, loss is {:.4f}.".format(test_acc,test_loss))
    return test_acc/100





# Get dataset
train_loader,test_loader,unlabeled_loader=getDataset()

# Get model
teacher_model_path=".\pre trained models\teacher.pkl"
studen1_path=".\pre trained models\studentpan.pkl"
student2_path=".\pre trained models\studentms.pkl"
teacher_model=torch.load(teacher_model_path).cuda()
student1_model=architecture.Student1_upgraded_version1().cuda()
student2_model=architecture.Student2().cuda()

# Hyparameters
learning_rate=0.001
EPO=200
# split_ratio=0.6
print("Initial TAE, testing teacher.")
TAE=TAS.Test_Teacher(teacher_model,test_loader)
th=0.1

optimizer1=optim.Adam(student1_model.parameters(),lr=learning_rate)
optimizer2=optim.Adam(student2_model.parameters(),lr=learning_rate)

# Get filtered data and Training
for epo in range(1,EPO+1):
    print("epoch is {}".format(epo))
    if epo==1:
        Acc_1,Acc_2=TAS.Test_Sudents(student1_model,student2_model,
                                    test_loader=test_loader,TAE=TAE)
    else:
        pass
    S_t=(1/2)*(Acc_1+Acc_2)
    DI=TAS.GenDI(th,type=0)
    print("Current S_t is {:.4f}, DI is {:.4f}".format(S_t,DI))
    if epo==1:
        pass
    else:
        TAE=TAS.updata_TAE(DI,epo,EPO,BaseLine=0.9,S_t=S_t,TAE=TAE)
    
    S=TAS.GenS(Acc_1,Acc_2,TAE)
    print("Score is {:.4f}".format(S))
    th,flag=TAS.updata_th(th,epo,EPO,S)
    
    ms_data,pan_data,label_data,dm,dp,dl=TAS.KLP(teacher_model=teacher_model,training_data=train_loader,th=th)
    filter_basic_loader=GetSpecialDataset(ms_data,pan_data,label_data)
    filter_diffi_loader=GetSpecialDataset(dm,dp,dl)
    print("unlabeled training, epoch is {}".format(epo))
    unlabeled_train(student1_model,student2_model,teacher_model,unlabeled_loader,optimizer1,optimizer2)
    print("labeled training, epoch is {}".format(epo))
    labeled_train(student1_model,student2_model,teacher_model,train_loader,optimizer1,optimizer2,DI,epo,EPO,flag)
    print("Testing, epoch is {}".format(epo))
    Acc_1,Acc_2,Loss_1,Loss_2 = Testing_GSN(student1_model,student2_model,test_loader)
    # if epo==EPO:
    m_pan,m_ms=test_second(student1_model,student2_model,test_loader)
    print("OA,AA,Kappa as follow")
    aa_oa(m_pan)
    aa_oa(m_ms)

if os.path.exists(studen1_path)==0:
    torch.save(student1_model,studen1_path)
else:
    pass
if os.path.exists(student2_path)==0:
    torch.save(student2_model,student2_path)
else:
    pass
