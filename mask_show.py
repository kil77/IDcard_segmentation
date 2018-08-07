import os
from PIL import Image

IMG_PATH = './datasets/IDImage/JPGImage'
PREDICT_PATH = './datasets/IDImage/inference_out'
ids = os.listdir(PREDICT_PATH)
for id in ids:
    id = id.split('.')[0]
    img_path = './datasets/IDImage/JPGImage/{}.jpg'.format(id)
    label_path = './datasets/IDImage/inference_out/{}.png'.format(id)
    label = Image.open(label_path)
    img = Image.open(img_path)
    label_px = label.load()
    img_px = img.load()
    width = img.size[0]
    higet = img.size[1]
    for w in range(width):
        for h in range(higet):
            if label_px[w, h] == (253, 231, 36, 255):
                a, b, c = img_px[w, h]
                img_px[w, h] = (0, b, c, 255)
    img.show()


