import torch
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
from torch import nn

# super parameters
batch_size = 64 # 每次模型更新参数时用到的样本数目
learning_rate = 0.01 # 控制模型参数更新的步长大小
momentum = 0.5 # SGD中的冲量
EPOCH = 10   # 遍历整个训练数据集的次数

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
transform = transforms.Compose([
    transforms.Resize((32, 32)), # 将图片大小调整为需要的大小以匹配网络输入
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])
train_dataset = datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)  
test_dataset = datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)  # train=True训练集，=False测试集
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # 载入数据集
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()


        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            #inplace-选择是否进行覆盖运算
            #意思是是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            #意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，
            #这样能够节省运算内存，不用多存储其他变量

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32    32*32*64
            #Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，
            # 一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,stride=2)   #(32-2)/2+1=16         16*16*64
        )


        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)     #(8-2)/2+1=4      4*4*256
        )

        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #(4-2)/2+1=2     2*2*512
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)   #(2-2)/2+1=1      1*1*512
        )


        self.conv=nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc=nn.Sequential(
            #y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            #nn.Liner(in_features,out_features,bias)
            #in_features:输入x的列数  输入数据:[batchsize,in_features]
            #out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            #bias: bool  默认为True
            #线性变换不改变输入矩阵x的行数,仅改变列数
            nn.Linear(512,512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256,10)
        )


    def forward(self,x):
        x=self.conv(x)
        #这里-1表示一个不确定的数，就是你如果不确定你想要reshape成几行，但是你很肯定要reshape成512列
        # 那不确定的地方就可以写成-1

        #如果出现x.size(0)表示的是batchsize的值
        # x=x.view(x.size(0),-1)
        x = x.view(-1, 512)
        x=self.fc(x)
        return x



# 初始化模型、损失函数和优化器 
model = Vgg16_net()
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  # 随机梯度下降更新参数，lr学习率，momentum冲量

def train(epoch):
    running_loss = 0.0  # 这整个epoch的loss清零
    total_loss = 0
    total_batches = len(train_loader)  # 获取总批次数

    # 遍历 train_loader 中的每个批次进行训练
    for batch_idx, data in enumerate(train_loader, 0):
        # 从数据中获取输入和目标标签
        inputs, target = data
        # 清除之前参数的梯度
        optimizer.zero_grad()

        # forward + backward + update
        # 用输入数据进行模型的前向传播，得到输出
        outputs = model(inputs)

        # 计算模型输出与目标标签之间的损失
        loss = criterion(outputs, target)

        # 执行反向传播和参数更新，以最小化损失
        loss.backward()
        optimizer.step()

        # 把运行中的loss累加起来，为了下面300次一除
        running_loss += loss.item()
        total_loss += loss.item()

        if batch_idx % 300 == 299:  # 不想要每一次都出loss，浪费时间，选择每300次出一个平均损失,和准确率
            print('[%d, %5d]: loss: %.3f'
                  % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0  # 这小批300的loss清零

    average_loss = total_loss / total_batches  # 计算平均损失
    print('[%d / %d]: loss on training set: %f ' % (epoch+1, EPOCH, loss))  # 求测试的准确率，正确数/总数
    return average_loss
        
    
def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 告诉 PyTorch 不需要计算梯度
        for data in test_loader:  # 遍历测试数据集中的每个批次
            images, labels = data
            outputs = model(images)
            # 预测的类别
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度，沿着行(第1个维度)去找1.最大值和2.最大值的下标
            total += labels.size(0)  # 总样本数
            # 张量之间的比较运算
            correct += (predicted == labels).sum().item()  # 总样本数中预测正确的个数
            
    acc = correct / total
    print('[%d / %d]: Accuracy on test set: %.1f %% ' % (epoch+1, EPOCH, 100 * acc))  # 求测试的准确率，正确数/总数
    return acc

if __name__ == '__main__':
    acc_list_test = []
    loss_list = []
    for epoch in range(EPOCH):
        loss = train(epoch)
        loss_list.append(loss)
        # if epoch % 10 == 9:  #每训练10轮 测试1次
        acc_test = test()
        acc_list_test.append(acc_test)
    # 保存模型参数
    torch.save(model, "./VGGNet_model.pth")

    # 绘制损失曲线
    plt.plot(loss_list, label='Training Loss')
    # 绘制准确率曲线
    plt.plot(acc_list_test, label='Accuracy On TestSet')
    
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()  # 添加图例
    # 保存图像
    plt.savefig('./visualization/loss_and_accuracy.png')
    plt.show()
