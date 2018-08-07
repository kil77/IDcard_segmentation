import numpy as np
import os
DATA_PATH = './datasets/IDImage'
IMG_PATH = './datasets/IDImage/JPGImage'
LABEL_PATH = './datasets/IDImage/Label'
def split():
    img_path = os.listdir(IMG_PATH)
    train_filename = os.path.join(DATA_PATH,'train.txt')
    val_filename = os.path.join(DATA_PATH,'val.txt')
    for item in img_path:
        item_id = item.split('.')[0]
        label = item_id + '.png'
        n = np.random.randint(0,10)
        if n < 8:
            if not os.path.exists(train_filename):
                os.system(r'touch %s' % train_filename)
            with open(train_filename,'a+') as train:
                train.write(os.path.join(IMG_PATH,item)+' '+os.path.join(LABEL_PATH,label) + '\n')
        else:
            if not os.path.exists(val_filename):
                os.system(r'touch %s' % val_filename)
            with open(val_filename,'a+') as val:
                val.write(os.path.join(IMG_PATH, item) + ' ' + os.path.join(LABEL_PATH, label) + '\n')



if __name__ == '__main__':
    split()