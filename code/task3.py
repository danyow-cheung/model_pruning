'''
20230729 剪枝实例： 多参数和全局剪枝
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


model = LeNet().to(device)
'''
多参数剪枝
'''
for name,module in model.named_modules():
    # 对所有conv2d的参数进行20%的l1非结构化剪枝
    if isinstance(module,torch.nn.Conv2d):
        prune.l1_unstructured(module,name='weight',amount=0.2)
	# 对所有linear层参数进行20%的l1非结构化剪枝
    elif isinstance(module,torch.nn.Linear):
	    prune.l1_unstructured(module,name='weight',amount=0.4)
print(dict(model.named_buffers()).keys())

'''
全局剪枝
通过删除整个模型最低的20%连结，而非删除每个层中最低20%连结 which mean 层与层之间删除比例不同
'''
model2 = LeNet().to(device)
parameters_to_prune = (
	(model.conv1,'weight'),
	(model.conv2,'weight'),
	(model.fc1,'weight'),
	(model.fc2,'weight'),
	(model.fc3,'weight'),
)

prune.global_unstructured(
	parameters_to_prune,
	pruning_method=prune.L1Unstructured,
	amount=0.2,
)
print(
    "稀疏性 in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    )
)
print(
    "稀疏性 in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv2.weight == 0))
        / float(model.conv2.weight.nelement())
    )
)
print(
    "稀疏性 in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc1.weight == 0))
        / float(model.fc1.weight.nelement())
    )
)
print(
    "稀疏性 in fc2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc2.weight == 0))
        / float(model.fc2.weight.nelement())
    )
)
print(
    "稀疏性 in fc3.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc3.weight == 0))
        / float(model.fc3.weight.nelement())
    )
)