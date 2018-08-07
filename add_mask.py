import cv2
from skimage.io import imread
from skimage import measure
from config import *

labels = os.listdir(args.output_path)
for label in labels:
    label_path = os.path.join(args.output_path,label)
    img = str(label.split('.')[0])+'.jpg'
    img_path = os.path.join(args.image_path,img)

    label_pic = imread(label_path)
    img_pic = cv2.imread(img_path)

    cv2.imshow('img',img_pic)
    cv2.waitKey(0)

    img_pic = img_pic.astype(np.float32)
    contours = measure.find_contours(label_pic,0.5)
    image = cv2.cvtColor(img_pic, cv2.COLOR_GRAY2BGR)
    for c in contours:
        c = np.around(c).astype(np.int)
        image[c[:, 0], c[:, 1]] = np.array((0,0,255))
    cv2.imshow('img_with_contour',img_pic)
    cv2.waitKey(0)


