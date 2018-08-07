from PIL import Image
import os

path = './dataset/IDImage'
json_path = os.listdir(path)
for item in json_path:
    item = os.path.join(path,item)
    if os.path.isdir(item):
        label_id = item.split('/')[-1].split('_')[0] + '.png'
        label_path = os.path.join(item,'label.png')
        new_label_path = os.path.join(path,label_id)
        os.rename(label_path,new_label_path)
