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
```python3 yolov5_pruning/train.py --data VOC.yaml --weights yolov5n.pt --img 640```

數據集選擇voc2007

### plan:
1. 先訓練yolov5n.pt 在voc.yaml的上結果
2. 再去對yolov5n.pt 模型做微調啥的
3. 