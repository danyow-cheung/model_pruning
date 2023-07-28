'''
20230728 剪枝实例：非结构化剪枝
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
print(model)
module = model.conv1 # 通过这一步得到conv1模块的参数和偏置数据
print('==============================')
print(list(module.named_parameters()))
print('缓存区数据\n',list(module.buffers()))

'''模块剪枝
'''
# 选用非结构化剪枝,选定conv1模块，剪枝比例是30%
prune.random_unstructured(module, name="weight", amount=0.3)

'''
'修剪的作用是将权重从参数中移除，并用一个名为weight_orig的新参数替换它(即在初始参数名称后面添加“_orig”)。
weight_trans存储了张量的未剪枝的版本。bias没有被修剪，所以它会保持不变。我们看看现在module的weight变成啥样了
'''
print('==============================')
print(list(module.named_parameters()))
# 由上面选择的剪枝技术生成的剪枝掩码被保存为名为 的模块缓冲区weight_mask（即附加"_mask"到初始参数name）。
print('缓存区数据\n',list(module.buffers()))

# 为了使前向传播无需修改即可工作，该weights属性需要存在。
print(module.weight)
# 最后，在使用 PyTorch 的每次前向传递之前应用剪枝 forward_pre_hooks。
# 具体来说，当被修剪时，正如我们在这里所做的那样，它将为与其相关的每个被修剪的参数module获取。
# forward_pre_hook在这种情况下，由于我们到目前为止只修剪了名为 的原始参数weight，因此只会出现一个钩子。

print(module._forward_pre_hooks)

'''同时也要对bias做剪枝'''
prune.l1_unstructured(module,name='bias',amount=3)
print(module.named_parameters())
