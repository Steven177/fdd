from transformers import DetrFeatureExtractor, DetrForObjectDetection
from PIL import Image
import requests
import torch
import torchvision
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def predict(im):
  # MODEL PREDICTION
  # AI models
  feature_extractor = DetrFeatureExtractor.from_pretrained('facebook/detr-resnet-50')
  model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

  # Forward pass
  encodings = feature_extractor(images=im, return_tensors="pt")
  outputs = model(**encodings)

  # model predicts bounding boxes and corresponding COCO classes
  logits = outputs.logits
  bboxes = outputs.pred_boxes

  # Select top classes
  probas = logits.softmax(-1)[0, :, :-1]
  keep = probas.max(-1).values > 0.9
  top_probas = probas[keep]
  # Select top bboxes
  target_sizes = torch.tensor(im.size[::-1]).unsqueeze(0)
  postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
  boxes = postprocessed_outputs[0]['boxes']
  top_boxes = postprocessed_outputs[0]['boxes'][keep]

  # Store in data format
  data = []
  for p, (xmin, ymin, xmax, ymax) in zip(top_probas, top_boxes.tolist()):
    cl = p.argmax()
    data.append({"score": p[cl].item(), "label": model.config.id2label[cl.item()], "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}})


  # EXPLANATION
  conv_features = []

  hooks = [
      model.model.backbone.conv_encoder.register_forward_hook(
          lambda self, input, output: conv_features.append(output)
      ),
  ]

  # propagate through the model
  outputs = model(**encodings, output_attentions=True)

  for hook in hooks:
      hook.remove()

  # don't need the list anymore
  conv_features = conv_features[0]
  # get cross-attention weights of last decoder layer - which is of shape (batch_size, num_heads, num_queries, width*height)
  dec_attn_weights = outputs.cross_attentions[-1]
  # average them over the 8 heads and detach from graph
  dec_attn_weights = torch.mean(dec_attn_weights, dim=1).detach()

  h, w = conv_features[-1][0].shape[-2:]
  count = 200
  for idx, (xmin, ymin, xmax, ymax) in zip(keep.nonzero(), top_boxes):
    fig = plt.figure()
    plt.imshow(dec_attn_weights[0, idx].view(h, w))
    plt.axis('off')
    fig.show()
    fig.savefig("media/explanations/explanation{}.png".format(count))
    count += 1
  return data




