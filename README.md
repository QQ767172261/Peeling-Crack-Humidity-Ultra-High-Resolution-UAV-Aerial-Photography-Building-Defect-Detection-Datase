无人机航拍建筑裂缝检测数据集-超高分辨率航拍建筑物缺陷检测数据集，2765张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a642f504ae544ec593f4fa7fd3fee029.png)

图像尺寸:3472*4624
类别数量:3类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/539d32e90c2f4542af705a00606bd4bb.png)

训练集图像数量:2419; 验证集图像数量:258； 测试集图像数量:88
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e9b570bddb6d4077a0be92a9f85ac930.png)

spalling - 剥落：541, 1026
crack - 裂缝：2561, 6853
moisture - 潮湿：248, 460
image num: 2765![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cfd32c2f26fa4d53bd9ba1e0eb52bdba.png)
# 超高分辨率航拍建筑物缺陷检测数据集
## 数据集信息表
|项目|详情|
| ---- | ---- |
|数据集名称|超高分辨率航拍建筑物缺陷检测数据集|
|图像总数量|2765张|
|原始图像尺寸|3472×4624（超高分辨率航拍）|
|标注格式|YOLO、VOC、COCO|
|类别总数|3类|
|训练集|2419张|
|验证集|258张|
|测试集|88张|

### 类别统计
|类别名称|中文释义|含该目标图像数|标注框总数|
| ---- | ---- | ---- | ---- |
|spalling|剥落|541|1026|
|crack|裂缝|2561|6853|
|moisture|潮湿|248|460|

## 数据集yaml配置文件 building_defect.yaml
```yaml
path: ./building_defect_dataset
train: images/train
val: images/val
test: images/test

names:
  0: spalling
  1: crack
  2: moisture
```

## 完整训练代码 train.py
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载yolo11n模型，可替换 yolo11s/yolo11m
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="building_defect.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=4,
        patience=15,
        cos_lr=True,
        amp=True,
        mosaic=1.0,
        mixup=0.1,
        save=True,
        project="runs/train",
        name="building_defect_detect"
    )
```

## 推理脚本 predict.py
```python
from ultralytics import YOLO

model = YOLO("./runs/train/building_defect_detect/weights/best.pt")

#单张超高分辨率图片推理
res = model.predict("test.jpg", conf=0.25, imgsz=640, save=True)

#批量推理
# res = model.predict("./test_images", conf=0.25, imgsz=640, save=True)
```

## 环境依赖
```bash
pip install ultralytics opencv-python
```

