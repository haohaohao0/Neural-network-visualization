import torch
from torchsummary import summary
from torch import nn
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

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

model = torch.load('./VGGNet_model.pth')


model_weights = []   # append模型的权重
conv_layers = []   # append模型的卷积层本身

# get all the model children as list
model_children = list(model.children())

# counter to keep count of the conv layers
counter = 0  # 统计模型里共有多少个卷积层

# print(model_children[0][0].weight)

# append all the conv layers and their respective wights to the list
for i in range(len(model_children)):  # 遍历最表层(Sequential就是最表层)
    if type(model_children[i]) == nn.Conv2d:   # 最表层只有一个卷积层
        counter+=1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])

    elif type(model_children[i]) == nn.Sequential:
        for child in model_children[i]:
            if type(child) == nn.Conv2d:
                counter+=1
                model_weights.append(child.weight)
                conv_layers.append(child)
print(f"Total convolution layers: {counter}")

outputs = []
names = []
image = Image.open('./1.jpg')
transform = transforms.Compose([
    transforms.Resize((32, 32)), # 将图片大小调整为需要的大小以匹配网络输入
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])
image = image.convert('L')
image = transform(image)
# print(f"Image shape before: {image.shape}")
image = image.unsqueeze(0)
# print(f"Image shape after: {image.shape}")


for layer in conv_layers[0:]:    # conv_layers即是存储了所有卷积层的列表
    image = layer(image)         # 每个卷积层对image做计算，得到以矩阵形式存储的图片，需要通过matplotlib画出
    outputs.append(image)
    names.append(str(layer))
# print(len(outputs))

# for feature_map in outputs:
#     print(feature_map.shape)

# print(outputs[1].shape)   
# print(outputs[1].squeeze(0).shape)   # 去掉 batch_size 的维度,因为matplotlib绘画，这个第0维没用
# print(torch.sum(outputs[1].squeeze(0),0).shape)   # 再次 .squeeze 将颜色通道这个维度去除, sum是把几十上百张灰度图像映射到一张

processed = []

for feature_map in outputs:
    feature_map = feature_map.squeeze(0)  # torch.Size([1, 64, 112, 112]) —> torch.Size([64, 112, 112])  去掉第0维 即batch_size维
    gray_scale = torch.sum(feature_map,0) # sum是把几十上百张灰度图像映射到一张,从彩色图片变为黑白图片  压缩64个颜色通道维度，否则feature map太多张
    gray_scale = gray_scale / feature_map.shape[0]  # 除以通道数求平均值 
    processed.append(gray_scale.data.cpu().numpy())  # .data是读取Variable中的tensor  .cpu是把数据转移到cpu上  .numpy是把tensor转为numpy

# for fm in processed:
#     print(fm.shape)


fig = plt.figure(figsize=(30, 50))

for i in range(len(processed)):   # len(processed) = 17
    a = fig.add_subplot(5, 4, i+1)
    img_plot = plt.imshow(processed[i])
    a.axis("off")  # 关闭子图 a 的坐标轴显示
    a.set_title(f'Conv2d-{i+1}', fontsize=30)   # names[i].split('(')[0] 结果为Conv2d

plt.savefig('./visualization/feature_maps.jpg', bbox_inches='tight')  # 若不加bbox_inches='tight'，保存的图片可能不完整