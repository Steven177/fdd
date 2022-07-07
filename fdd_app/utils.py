import json
import numpy as np
import torch
from torchvision.ops.boxes import _box_inter_union
from torchvision.ops.boxes import box_area
import requests

API_TOKEN = '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"

coco_distribution = ["person", "bicycle", "car","motorcycle","airplane","bus","train","truck","traffic light","fire hydrant","stop sign", "parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

def query(filename):
  with open(filename, "rb") as f:
      data = f.read()
  response = requests.request("POST", API_URL, headers=headers, data=data)
  return json.loads(response.content.decode("utf-8"))


def padd_boxes(boxes_exp, boxes_pred):
  diff = boxes_exp.shape[0] - boxes_pred.shape[0]
  if diff > 0:
    boxes_pred = np.pad(boxes_pred, ((0,abs(diff)),(0,0)), 'constant')
  elif diff < 0:
    boxes_exp = np.pad(boxes_exp, ((0,abs(diff)),(0,0)), 'constant')
  return boxes_exp, boxes_pred


def calculate_l1_loss(exps, preds, sample):
  num_of_exps = len(exps)
  num_of_preds = len(preds)

  l1_cost_mat = np.zeros((num_of_exps, num_of_preds))

  for i, exp in enumerate(exps):
    for j, pred in enumerate(preds):
      # convert boxes
      exp_box_xywh = [exp.xmin / sample.image.width, exp.ymin / sample.image.height, (exp.xmax - exp.xmin) / sample.image.width, (exp.ymax - exp.ymin) / sample.image.height]
      pred_box_xywh = [pred.xmin/ sample.image.width, pred.ymin / sample.image.height, (pred.xmax - pred.xmin) / sample.image.width, (pred.ymax - pred.ymin) / sample.image.height]

      exp_box_xywh = np.matrix(exp_box_xywh)
      pred_box_xywh = np.matrix(pred_box_xywh)
      delta_box = abs(exp_box_xywh - pred_box_xywh)
      l1_cost_mat[i][j] = delta_box.sum()
  return l1_cost_mat

def calculate_box_loss(giou_cost_mat, l1_cost_mat, lamda_iou=0.5, lamda_l1=0.5):
  return lamda_iou * giou_cost_mat + lamda_l1 * l1_cost_mat

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
  area1 = box_area(boxes1)
  area2 = box_area(boxes2)

  lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
  rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

  wh = (rb - lt).clamp(min=0)  # [N,M,2]
  inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

  union = area1[:, None] + area2 - inter

  iou = inter / union
  return iou, union


def generalized_box_iou_loss(exps, preds):
  """
  Generalized IoU from https://giou.stanford.edu/
  The boxes should be in [x0, y0, x1, y1] format
  Returns a [N, M] pairwise matrix, where N = len(boxes1)
  and M = len(boxes2)
  """
  boxes1 = []
  for exp in exps:
    boxes1.append([exp.xmin, exp.ymin, exp.xmax, exp.ymax])
  boxes2 = []
  for pred in preds:
    boxes2.append([pred.xmin, pred.ymin, pred.xmax, pred.ymax])

  # numpy -> pytroch tensor
  boxes1 = torch.Tensor(boxes1)
  boxes2 = torch.Tensor(boxes2)
  # degenerate boxes gives inf / nan results
  # so do an early check
  assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
  assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
  iou, union = box_iou(boxes1, boxes2)

  lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
  rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

  wh = (rb - lt).clamp(min=0)  # [N,M,2]
  area = wh[:, :, 0] * wh[:, :, 1]

  giou = iou - (area - union) / area
  loss = 1 - giou

  return loss.numpy()

def padd_labels(labels_exp, labels_pred):
  diff = len(labels_exp) - len(labels_pred)
  if diff > 0:
    for i in range(abs(diff)):
      labels_pred.append(["no object"])
  elif diff < 0:
    for i in range(abs(diff)):
      labels_exp.append(["no object"])
  return labels_exp, labels_pred

def calculate_class_loss(exps, preds):
  num_of_exp = len(exps)
  num_of_pred = len(preds)

  labels_cost_matrix = np.ones((num_of_exp, num_of_pred))
  for i, exp in enumerate(exps):
    for j, pred in enumerate(preds):
      if exp.label == pred.label:
        labels_cost_matrix[i][j] = 0
  return labels_cost_matrix

def calculate_matching_loss(loss_box, loss_labels, weight_box=0.5, weight_labels=0.5):
  return weight_box * loss_box + weight_labels * loss_labels

def check_if_match(exp_label, pred_label):
  return False if exp_label == "no object" or pred_label == "no object" else True

def check_if_indistribution(exp_label):
  for i in coco_distribution:
    if exp_label.lower() == i:
      return True
  return False

def check_quality_of_box(exp, pred , threshold=0.7):
  iou = calculate_iou(exp, pred)
  return True if iou <= threshold else False

def calculate_iou(exp, pred):
  """
  assert exp_box["xmin"] < exp_box["xmax"]
  assert exp_box["ymin"] < exp_box["ymax"]
  assert pred_box["xmin"] > pred_box["xmax"]
  assert pred_box["ymin"] > pred_box["ymax"]
  """
  # determine the coordinates of the intersection rectangle
  x_left = max(exp.xmin, pred.xmin)
  y_top = max(exp.ymin, pred.ymin)
  x_right = min(exp.xmax, pred.xmax)
  y_bottom = min(exp.ymax, pred.ymax)

  if x_right < x_left or y_bottom < y_top:
    return 0.0

  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  exp_box_area = (exp.xmax - exp.xmin) * (exp.ymax - exp.ymin)
  pred_box_area = (pred.xmax - pred.xmin) * (pred.ymax - pred.ymin)

  iou = intersection_area / float(exp_box_area + pred_box_area - intersection_area)
  return iou

def check_quality_of_score(score, threshold=0.95):
  return True if score <= threshold else False


