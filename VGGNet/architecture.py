
import sys
sys.path.append('../')
from pycore.tikzeng import *

# defined your arch
arch = [
    to_head( '..' ), # 生成.tex文件位置
    to_cor(),
    to_begin(),
     #input
    to_input( './1.jpg',to='(-10,0,0)', width=30, height=30),   # 显示输入图像，比较美观
    # 该层的图像大小， 输出通道大小， 表示这一层与上一层分别在x，y，z上的偏移量，一般只需要调整x，表示该层在x，y，z方向上的坐标
    # 这一层在pool1层的右边，后三个都是视觉效果 
    to_ConvConvRelu(name='layer1', s_filer=32, n_filer=(64, 64), offset="(0,0,0)", to="(0,0,0)", width=(5, 5), height=120, depth=120,caption='layer1'),
    to_Pool("pool1", offset="(0,0,0)",to="(layer1-east)",width=1, height=60, depth=60,opacity=0.5),
    
    to_ConvConvRelu(name='layer2', s_filer=16, n_filer=(128, 128), offset="(10,0,0)", to="(0,0,0)", width=(5, 5), height=100, depth=100,caption='layer2'),
    to_Pool("pool2", offset="(0,0,0)",to="(layer2-east)",width=1, height=50, depth=50,opacity=0.5),
    to_connection("pool1", "layer2"),
    
    to_ConvConvRelu("layer3", n_filer=(256, 256), offset="(10,0,0)", to="(pool2-east)", height=80, depth=80, width=(4,4),caption="layer3" ),
    to_ConvConvRelu("layer3-1", n_filer=(256, 256), offset="(0,0,0)", to="(layer3-east)", height=80, depth=80, width=(4,0) ),
    to_Pool("pool3", offset="(0,0,0)",to="(layer3-1-east)",width=1,height=40, depth=40,opacity=0.5),
    to_connection("pool2", "layer3"),

    to_ConvConvRelu("layer4", s_filer=4, n_filer=(512, 512), offset="(10,0,0)", to="(pool3-east)", height=60, depth=60, width=(4,4),caption="layer4" ),
    to_ConvConvRelu("layer4-1", s_filer=4, n_filer=(512, 512), offset="(0,0,0)", to="(layer4-east)", height=60, depth=60, width=(4,0) ),
    to_Pool("pool4", offset="(0,0,0)",to="(layer4-1-east)",width=1,height=30, depth=30,opacity=0.5),
    to_connection("pool3", "layer4"),
    
    to_ConvConvRelu("layer5", s_filer=2, n_filer=(512, 512), offset="(10,0,0)", to="(pool4-east)", height=40, depth=40, width=(4,4),caption="layer5" ),
    to_ConvConvRelu("layer5-1", s_filer=2, n_filer=(512, 512), offset="(0,0,0)", to="(layer5-east)", height=40, depth=40, width=(4,0) ),
    to_Pool("pool5", offset="(0,0,0)",to="(layer5-1-east)",width=1,height=20, depth=20,opacity=0.5),
    to_connection("pool4", "layer5"),
    
    to_Conv("fc1", 512, 512, offset="(5,0,0)", to="(pool5-east)", height=30, depth=30, width=30 ),
    to_connection("pool5", "fc1"),
    to_Conv("fc2", 512, 256, offset="(5,0,0)", to="(fc1-east)", height=30, depth=30, width=30 ),
    to_connection("fc1", "fc2"),
    to_Conv("fc3", 256, 10, offset="(5,0,0)", to="(fc2-east)", height=30, depth=30, width=30 ),
    to_connection("fc2", "fc3"),
    to_SoftMax("soft1", 10 ,"(3,0,0)", "(fc3-east)", caption="SOFT"  ),
    to_connection("fc3", "soft1"),
    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()