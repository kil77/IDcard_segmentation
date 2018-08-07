import tensorflow as tf
from scipy import misc
def random_crop_and_resize_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Input:
    image: [1, height, width, 3] uint8
    gt_mask: [1, height, width, 1] uint8

    Return:
    image: [height, width, 3] float32
    gt_mask: [height, width, 1] uint8
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=3, values=[image, label]) 
    combined_shape = tf.shape(combined)

    nh = tf.to_int32(tf.maximum(crop_h, combined_shape[1]))
    nw = tf.to_int32(tf.maximum(crop_w, combined_shape[2]))
    combined_pad = tf.image.resize_bilinear(combined, [nh, nw])

    # combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[1]), tf.maximum(crop_w, image_shape[2]))
    
    last_image_dim = tf.shape(image)[-1]
    combined_pad = tf.squeeze(combined_pad, [0])
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop

def flip_image_w(image):
    """
    image: [height, width, 3]
    """
    return tf.reverse(image, axis=[1])

def flip_image_h(image):
    """
    image: [height, width, 3]
    """
    return tf.reverse(image, axis=[0])

def rescale(image, gt_mask, h, w, scale):
    """
    Input:
    image: [height, width, 3] uint8
    gt_mask: [height, width, 1] uint8

    Return:
    image: [1, height, width, 3] float32
    gt_mask: [1, height, width, 1] uint8
    """
    image = tf.to_float(image)
    image = tf.expand_dims(image, 0)
    gt_mask = tf.expand_dims(gt_mask, 0)
    nh = tf.to_int32(tf.to_float(h) * scale)
    nw = tf.to_int32(tf.to_float(w) * scale)
    new_image = tf.image.resize_bilinear(image, [nh, nw])
    new_mask = tf.image.resize_nearest_neighbor(gt_mask, [nh, nw])
    return new_image, new_mask

def rotate_image(image):
    """
    Input:
    image: [height, width, 3] uint8
    gt_mask: [height, width, 1] uint8

    Return:
    image: [height, width, 3] uint8
    gt_mask: [height, width, 1] uint8
    """