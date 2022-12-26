import torch
import argparse

import sys
print(sys.path)

import util_funcs.model as ml
from PIL import Image
import torchvision.transforms as T
from datetime import date, datetime
import matplotlib.pyplot as plt



# parser = argparse.ArgumentParser()
# parser.add_argument('model', nargs='?', default="model", help="Path to the model")
# args = parser.parse_args()

model_path = "/home/weixuan/Documents/Code/blenderproc/model/model_2022_12_26_14_21_21.pt"

num_classes = 4
model = ml.get_object_detector_model(num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()

#visualization
# img_path = "/home/weixuan/Documents/Code/blenderproc/data/annotation/images/000019.jpg"
img_path = "/home/weixuan/Documents/Code/blenderproc/target_images/002.png"
imgs = Image.open(img_path)
transform = T.Compose([T.ToTensor()])
imgs = transform(imgs)
predictions = model([imgs])

#save trained model
now = datetime.now()
today = date.today()
today_str = today.strftime("%Y_%m_%d")
current_time = now.strftime("%H_%M_%S")
time = today_str + "_" + current_time
model_name = "model_" + time  + ".pt"
# save_model_file_path = "/home/weixuan/Documents/Code/blenderproc/model/" + model_name
# torch.save(model.state_dict(), save_model_file_path)
# print("saved model to {}".format(save_model_file_path))

class_names = [ '__background__', 'can', 'bottle', 'N/A']
confidence=0.5
rect_th=2
text_size=1
text_th=1
pred_class = [class_names[i] for i in list(predictions[0]['labels'].numpy())]
pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(predictions[0]['boxes'].detach().numpy())]
pred_score = list(predictions[0]['scores'].detach().numpy())
pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
pred_boxes = pred_boxes[:pred_t+1]
pred_class = pred_class[:pred_t+1]

import cv2
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# print(len(boxes))
for i in range(len(pred_boxes)):
    cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=(0, 255, 0), thickness=rect_th)
    cv2.putText(img,pred_class[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
plt.figure(figsize=(20,30))
plt.imshow(img)
plt.xticks([])
plt.yticks([])
plt.show()
