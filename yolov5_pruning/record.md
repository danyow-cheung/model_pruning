# 根據所學的模型減枝的技術，來對yolov5s進行模型剪枝

## 訓練代碼
**激活虛擬環境**
```source /Users/danyow/Envs/yolo_env/bin/activate```


**開始訓練**
> 1. yolov5n的內存更小，更適合乞丐版macos
> 2. VOC.yaml 數據集更少
下載數據集代碼
```python3 yolov5_pruning/train.py --data VOC.yaml ``` 
訓練代碼
```python3 yolov5_pruning/train.py --data VOC.yaml --weights pretrained_model/yolov5n.pt --img 320```

數據集選擇voc2007

### plan:
1. 先訓練yolov5n.pt 在voc.yaml的上結果
   > 使用colab來跑，相關配置是yolov5s.pt img640
   > 更多配置是看opt.yaml，主要內容在跑colab.ipynb

2. 再去對yolov5n.pt 模型做微調啥的
   > 先使用torch官方的剪枝來操作
   
   research一下發現，常見的剪枝方法有 
   - https://github.com/ultralytics/yolov5/issues/304
   - https://medium.com/nerd-for-tech/how-to-prune-sparse-yolov5-da19e1d84a6
   直接就是`prune(model,0.3)`

   目前想做的事，隨意加減模型（使模型能跑不跑錯），和原始模型做map的對比

   20230807 實現非結構化剪枝，不跑錯現在要去學習，剪枝的策略
   
   20230808 没啥好学的哈哈哈哈,看实际情况吧。大致流程通了。
   
