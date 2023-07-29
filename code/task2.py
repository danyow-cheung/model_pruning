'''
20230729 剪枝实例：结构化剪枝
'''
import torch 
from torch import nn 
import torch.nn.utils.prune as prune 
import torch.nn.functional as F

device = torch.device("cuda"if torch.cuda.is_available()else 'cpu')

'''构建LENET网络'''
class LeNet(nn.Module):
	def __init__(self):
		super(LeNet,self).__init__()
		# 单通道图像输入。5x5核尺寸	
		self.conv1 = nn.Conv2d(1,3,5)
		self.conv2 = nn.Conv2d(3,16,5)
		self.fc1 = nn.Linear(16*5*5,120)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self,X):
		x = F.max_pool2d(F.relu(self.conv1(X)),(2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)),2)
		x = x.view(-1,int(x.nelement()/x.shape[0]))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x 


'''获取网络需要剪枝的模块'''
model = LeNet().to(device)
module = model.conv1
print(list(module.named_parameters()))



# 根据通道的L2范数，沿着张量的第0轴对weight参数进行结构化剪枝
# 使用`ln_structured`方法，剪枝比例为33
prune.ln_structured(module,name='weight',amount=0.33,n=2,dim=0)
print(module.weight)
print('-'*49)
'''所有的张量，包括mask缓存区和用于计算剪枝张量的原始参数都保存在模型的state_dict中'''
print(model.state_dict().keys())
# 要使修建always的话，可以删除weight_trans和weight_mask重新参数化，并删除forward_pre_hook
# 使用`torch.nn.util.prune.remove`函数可以做到`
prune.remove(module,'weights')
