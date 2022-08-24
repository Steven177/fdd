import replicate
import os
from PIL import Image

clip_prefix = replicate.models.get("rmokady/clip_prefix_caption")
clip_caption = replicate.models.get("j-min/clip-caption-reward")
output = clip_prefix.predict(image="...")


image_captioner = replicate.models.get("salesforce/blip")
path = "woz1copy.jpeg"
print(path)
print(os.getcwd())
output = image_captioner.predict(image=open("img.jpeg", "rb"))
print("done")
print(output)

for o in output:
  print(o)
  print(o["text"])
  print(o["text"].split(": ")[1])

# crop
left = 10
top = 10
right = 200
bottom = 200
pil_image_obj = Image.open("img.jpeg")
im = pil_image_obj.crop([left, top, right, bottom])
im.show()

