# TOG-Vanishing-Attacks-on-YOLO11
TOG: Targeted Adversarial Objectness Gradient Attacks on Real-time Object Detection Systems

Задание посвящено атаке нейронной сети, решающей задачу детекции (обнаружения) объектов на изображениях.
  Normal Detection  |  TOG-Vanishing
:-------------------------:|:-------------------------:
![]([https://github.com/Cuperboy/TOG-Vanishing-Attacks-on-YOLO11/blob/main/outputs/yolo_people.gif](https://github.com/Pavelslavovich/Pavelslavovich-TOG-Vanishing-Attacks-on-YOLO11/blob/9e90c79588180683afbf73e8351a44636ac13c77/Result/flex_predict.gif))  |  ![]([https://github.com/Cuperboy/TOG-Vanishing-Attacks-on-YOLO11/blob/main/outputs/attacked_people.gif](https://github.com/Pavelslavovich/Pavelslavovich-TOG-Vanishing-Attacks-on-YOLO11/blob/9e90c79588180683afbf73e8351a44636ac13c77/Result/attacked_video.gif))



Две недели назад вышла новая версия самой популярной сети для детекции — YOLOv11. Именно была взята в качестве жертвы.
YOLOv11 была обучена на наборе данных COCO 2017. Поэтому и оценка работы нейросети в условиях атаки была произведена на этом наборе. Валидационная часть набора COCO 2017 состоит из 5000 изображений с верной (ground truth) разметкой.

Цель задания — реализовать такую состязательную атаку на YOLOv11, которая приведет к исчезновению рамок с объектами в предсказаниях модели на атакованных картинках.
Алгоритм атаки приведен в статье [TOG: Targeted Adversarial Objectness Gradient Attacks on Real-time Object Detection Systems](https://arxiv.org/abs/2004.04320) и называется TOG-vanishing. 

Были использованы следующие гиперпараметры атаки:
- Linf-норма вносимого в картинки возмущения — 4/255;
- размер шага в направлении градиента — 1/255;
- количество итераций атаки — 10.
