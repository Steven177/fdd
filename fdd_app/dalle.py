"""
# API
export REPLICATE_API_TOKEN=<token>
import replicate
model = replicate.models.get("kuprel/min-dalle")
model.predict(text="Dali painting of WALL·E")
"""

"""
# MANUAL IMPORT
from min_dalle import MinDalle
import torch
import numpy
import requests

def load_dalle():
  model = MinDalle(
      models_root='./pretrained',
      dtype=torch.float16,
      is_mega=True,
      is_reusable=True
  )
  dalle_loaded = True
  return model, dalle_loaded

def create_image(query):
  images = model.generate_images(
    text= query,
    seed=-1,
    image_count=7,
    log2_k=6,
    log2_supercondition_factor=5,
    is_verbose=False
  )

  images = images.to('cpu').numpy()
  return images


def call_dalle(query):
  model = replicate.models.get("kuprel/min-dalle")
  return model.predict(text=query)

"""

