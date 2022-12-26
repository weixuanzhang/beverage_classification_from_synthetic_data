import os
import torch
import torch.utils.data
from PIL import Image
from pycocotools.coco import COCO
from torch.utils import data
from torchvision import transforms
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from datetime import date, datetime
import argparse
# from sklearn.model_selection import train_test_split



parser = argparse.ArgumentParser()
parser.add_argument('model', nargs='?', default="model", help="Path to the model")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class myOwnDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]['file_name']
        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        #TODO: weixuan, use the category id as the labels
        labels = torch.as_tensor([coco_annotation[i]['category_id'] for i in range(num_objs)],dtype=torch.int64)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform():
    return T.Compose([T.ToTensor()])


#load the data
root = "/home/weixuan/Documents/Code/blenderproc/data/annotation"
filenames = ['000000.jpg']
labels = [0]

train_data_dir = root
train_coco = "/home/weixuan/Documents/Code/blenderproc/data/annotation/coco_annotations.json"
my_coco_dataset = myOwnDataset(root=train_data_dir,
                          annotation=train_coco,
                          transforms=get_transform()
                          )

batch_size = 1
num_workers = 4
data_loader = torch.utils.data.DataLoader(my_coco_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=num_workers,
                                         collate_fn=collate_fn)

image_datasets = datasets.ImageFolder(root)
dataset_sizes = len(image_datasets)

#get model
num_classes = 4
def get_object_detector_model(num_classes, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model.to(device)

#set up training
import time
import copy
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        model.train()
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for inputs, labels in data_loader:
            # inputs = inputs.to(device)
            # labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(outputs == labels.data)
                scheduler.step()

                epoch_loss = running_loss / dataset_sizes
                epoch_acc = running_corrects.double() / dataset_sizes

                print(f' Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model = get_object_detector_model(num_classes, device)

from util_funcs.engine import train_one_epoch, evaluate

criterion = nn.CrossEntropyLoss()

# Observe that only parameters of final layer are being optimized as
# opposed to before.
optimizer_conv = optim.SGD(model.roi_heads.box_predictor.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

num_epochs = 15
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer_conv, data_loader, device, epoch, print_freq = 1)

# model_conv = train_model(model, criterion, optimizer_conv,
#                          exp_lr_scheduler, num_epochs=5)


#save trained model
now = datetime.now()
today = date.today()
today_str = today.strftime("%Y_%m_%d")
current_time = now.strftime("%H_%M_%S")
time = today_str + "_" + current_time
model_name = "model_" + time  + ".pt"
save_model_file_path = "/home/weixuan/Documents/Code/blenderproc/model/" + model_name
torch.save(model.state_dict(), save_model_file_path)
print("saved model to {}".format(save_model_file_path))
print("Training done!")

# # test object detector and  inference
# images, targets = next(iter(data_loader))
# images = list(image for image in images)
# targets = [{k: v for k, v in t.items()} for t in targets]
# output = model(images,targets)   # Returns losses and detections
#
# model.eval()
#
# img_path = "/home/weixuan/Documents/Code/blenderproc/data/annotation/images/000017.jpg"
# imgs = Image.open(img_path)
# transform = T.Compose([T.ToTensor()])
# imgs = transform(imgs)
# predictions = model([imgs])
#
# class_names = [ '__background__', 'can', 'bottle', 'N/A']
# confidence=0.5
# rect_th=2
# text_size=1
# text_th=1
# pred_class = [class_names[i] for i in list(predictions[0]['labels'].numpy())]
# pred_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(predictions[0]['boxes'].detach().numpy())]
# pred_score = list(predictions[0]['scores'].detach().numpy())
# pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
# pred_boxes = pred_boxes[:pred_t+1]
# pred_class = pred_class[:pred_t+1]
#
# import cv2
# img = cv2.imread(img_path)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # print(len(boxes))
# for i in range(len(pred_boxes)):
#     cv2.rectangle(img, pred_boxes[i][0], pred_boxes[i][1],color=(0, 255, 0), thickness=rect_th)
#     cv2.putText(img,pred_class[i], pred_boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
# plt.figure(figsize=(20,30))
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# plt.show()
