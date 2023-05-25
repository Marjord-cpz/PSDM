
import architecture
from getDataset import getDataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import os

def Train_Teacher(teacher,train_loader,optimizer):
    teacher.train()
    correct=0.0
    for step,(ms,pan,label,_) in enumerate(tqdm(train_loader)):
        ms,pan,label=ms.cuda(),pan.cuda(),label.cuda()
        # print("ms",ms.size())
        fusion_data=architecture.IHS(ms,pan)
        teacher_output=teacher(fusion_data)
        loss=F.cross_entropy(teacher_output,label.long())
        pred_teacher=teacher_output.max(1,keepdim=True)[1]
        # print("output:",teacher_output.size())
        # print("pred_teacher:",pred_teacher.size())
        # print("label:",label.size())
        temp=pred_teacher.eq(label.view_as(pred_teacher).long())
        score_test=F.softmax(teacher_output,dim=1)
        score_test=torch.nonzero(score_test>0.9)
        # print("temp:",temp.squeeze())
        # print("temp:",temp)
        # print("filter:",score_test.size())
        correct+=pred_teacher.eq(label.view_as(pred_teacher).long()).sum().item()
        index_test=torch.tensor([1,2])
        # print("pan",pan.size())
        # print("pan_index",pan[index_test].size())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    Train_correct=correct*100/len(train_loader.dataset)
    print("Train Acc: {:.4f}\t Train Loss: {:.6f} (step is {})".format(Train_correct,loss,step+1))
    pass

def Test_Teacher(teacehr,test_loader):
    teacehr.eval()
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
    print("Test Acc: {:.4f}\t Test Loss: {:.6f}".format(Test_Acc,Test_Loss))
    pass



# hyparameters
learning_rate=0.001
EPOCH=10
# loading data
train_loader,test_loader,unlabel_loader=getDataset()
# loading model
teacher=architecture.Teacher().cuda()
optimizer=optim.Adam(teacher.parameters(),lr=learning_rate)

# training and testing
for epoch in  range(1,EPOCH+1):
    print("======Training, epoch is {}======".format(epoch))
    Train_Teacher(teacher,train_loader,optimizer)
    print("======Testing======")
    Test_Teacher(teacher,test_loader)

# save model
teacher_path=".\pre trained models\teacher.pkl"
if os.path.exists(teacher_path)==0:
    torch.save(teacher,teacher_path)
else:
    pass





