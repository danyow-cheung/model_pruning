# 实例篇

## 非结构化剪枝
> task1.py
### pytorch中buffer和parameter的区别
一般来说，torhc模型中需要保存的参数包括两种
- 反向传播需要被编译器optimizer更新的，称呼为parameter
- 与之对应，同样在反向传播但不需要更新的，称呼为buffer

### 剪枝核心
剪枝一个模块，需要三步：
1. 在torch.nn.utils.prune中选定一个剪枝方案，或者自定义(通过子类BasePruningMethod)
2. 指定需要剪枝的模块和对应的名称
3. 输入对应函数需要的参数


## 结构化剪枝
> task2.py
### 剪枝核心
和上同

## 多参数和全局剪枝
> task3.py
