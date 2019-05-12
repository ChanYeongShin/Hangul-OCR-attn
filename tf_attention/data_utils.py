__author__ = 'solivr'
__license__ = "GPL"


import tensorflow as tf
import numpy as np
from typing import Tuple

def random_rotation(img: tf.Tensor, max_rotation: float=0.1, crop: bool=True) -> tf.Tensor:  # adapted from SeguinBe
    """
    Rotates an image with a random angle.
    See https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae
    :param img: Tensor
    :param max_rotation: maximum angle to rotate (radians)
    :param crop: boolean to crop or not the image after rotation
    :return: rotated image
    """
    with tf.name_scope('RandomRotation'):
        rotation = tf.random_uniform([], -max_rotation, max_rotation, name='pick_random_angle')
        rotated_image = tf.contrib.image.rotate(img, rotation, interpolation='BILINEAR')
        if crop:
            rotation = tf.abs(rotation)
            original_shape = tf.shape(rotated_image)[:2]
            h, w = original_shape[0], original_shape[1]
            old_l, old_s = tf.cond(h > w, lambda: [h, w], lambda: [w, h])
            old_l, old_s = tf.cast(old_l, tf.float32), tf.cast(old_s, tf.float32)
            new_l = (old_l * tf.cos(rotation) - old_s * tf.sin(rotation)) / tf.cos(2*rotation)
            new_s = (old_s - tf.sin(rotation) * new_l) / tf.cos(rotation)
            new_h, new_w = tf.cond(h > w, lambda: [new_l, new_s], lambda: [new_s, new_l])
            new_h, new_w = tf.cast(new_h, tf.int32), tf.cast(new_w, tf.int32)
            bb_begin = tf.cast(tf.ceil((h-new_h)/2), tf.int32), tf.cast(tf.ceil((w-new_w)/2), tf.int32)
            # Test sliced
            rotated_image_crop = tf.cond(
                tf.logical_and(bb_begin[0] < h - bb_begin[0], bb_begin[1] < w - bb_begin[1]),
                true_fn=lambda: rotated_image[bb_begin[0]:h - bb_begin[0], bb_begin[1]:w - bb_begin[1], :],
                false_fn=lambda: img,
                name='check_slices_indices'
            )

            # If crop removes the entire image, keep the original image
            rotated_image = tf.cond(tf.equal(tf.size(rotated_image_crop), 0),
                                    true_fn=lambda: img,
                                    false_fn=lambda: rotated_image_crop,
                                    name='check_size_crop')

        return rotated_image


def random_padding(image: tf.Tensor, max_pad_w: int=5, max_pad_h: int=5) -> tf.Tensor:
    """
    Given an image will pad its border adding a random number of rows and columns
    :param image: image to pad
    :param max_pad_w: maximum padding in width
    :param max_pad_h: maximum padding in height
    :return: a padded image
    """

    w_pad = list(np.random.randint(0, max_pad_w, size=[2]))
    h_pad = list(np.random.randint(0, max_pad_h, size=[2]))
    paddings = [h_pad, w_pad, [0, 0]]

    return tf.pad(image, paddings, mode='REFLECT', name='random_padding')


def image_scaling(image: tf.Tensor, minscale: float=0.5, maxscale: float=1.5) -> tf.Tensor:
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.
    Args:
      image: Training image to scale.
      minscale: minimal scaling factor.
      maxscale: maximal scaling factor.
    """

    scale = tf.random_uniform([1], minval=minscale, maxval=maxscale, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]))
    scaled_image = tf.image.resize_images(image, new_shape)

    return scaled_image


def augment_data(image: tf.Tensor, max_rotation: float=0.1) -> tf.Tensor:
    """
    Data augmentation on an image (padding, brightness, contrast, rotation)
    :param image: Tensor
    :param max_rotation: float, maximum permitted rotation (in radians)
    :return: image: Tensor
    """
    with tf.name_scope('DataAugmentation'):

        # Random padding
        image = random_padding(image)

        image = image_scaling(image, minscale=0.5, maxscale= 1.5)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, 0.5, 1.5)
        image = random_rotation(image, max_rotation, crop=True)

        if image.shape[-1] >= 3:
            image = tf.image.random_hue(image, 0.2)
            image = tf.image.random_saturation(image, 0.5, 1.5)

        return image


def padding_inputs_width(image: tf.Tensor, target_shape: Tuple[int, int], increment: int) \
        -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Given an input image, will pad it to return a target_shape size padded image.
    There are 3 cases:
         - image width > target width : simple resizing to shrink the image
         - image width >= 0.5*target width : pad the image
         - image width < 0.5*target width : replicates the image segment and appends it
    :param image: Tensor of shape [H,W,C]
    :param target_shape: final shape after padding [H, W]
    :param increment: reduction factor due to pooling between input width and output width,
                        this makes sure that the final width will be a multiple of increment
    :return: (image padded, output width)
    """

    target_ratio = target_shape[1]/target_shape[0]
    # Compute ratio to keep the same ratio in new image and get the size of padding
    # necessary to have the final desired shape
    shape = tf.shape(image)
    ratio = tf.divide(shape[1], shape[0], name='ratio')

    new_h = target_shape[0]
    new_w = tf.cast(tf.round((ratio * new_h) / increment) * increment, tf.int32)
    f1 = lambda: (new_w, ratio)
    f2 = lambda: (new_h, tf.constant(1.0, dtype=tf.float64))
    new_w, ratio = tf.case({tf.greater(new_w, 0): f1,
                            tf.less_equal(new_w, 0): f2},
                           default=f1, exclusive=True)
    target_w = target_shape[1]

    # Definitions for cases
    def pad_fn():
        with tf.name_scope('mirror_padding'):
            pad = tf.subtract(target_w, new_w)

            img_resized = tf.image.resize_images(image, [new_h, new_w])

            # Padding to have the desired width
            paddings = [[0, 0], [0, pad], [0, 0]]
            pad_image = tf.pad(img_resized, paddings, mode='SYMMETRIC', name=None)

            # Set manually the shape
            pad_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_image, (new_h, new_w)

    def replicate_fn():
        with tf.name_scope('replication_padding'):
            img_resized = tf.image.resize_images(image, [new_h, new_w])

            # If one symmetry is not enough to have a full width
            # Count number of replications needed
            n_replication = tf.cast(tf.ceil(target_shape[1]/new_w), tf.int32)
            img_replicated = tf.tile(img_resized, tf.stack([1, n_replication, 1]))
            pad_image = tf.image.crop_to_bounding_box(image=img_replicated, offset_height=0, offset_width=0,
                                                      target_height=target_shape[0], target_width=target_shape[1])

            # Set manually the shape
            pad_image.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return pad_image, (new_h, new_w)

    def simple_resize():
        with tf.name_scope('simple_resize'):
            img_resized = tf.image.resize_images(image, target_shape)

            img_resized.set_shape([target_shape[0], target_shape[1], img_resized.get_shape()[2]])

            return img_resized, tuple(target_shape)

    # 3 cases
    pad_image, (new_h, new_w) = tf.case(
        {  # case 1 : new_w >= target_w
            tf.logical_and(tf.greater_equal(ratio, target_ratio),
                           tf.greater_equal(new_w, target_w)): simple_resize,
            # case 2 : new_w >= target_w/2 & new_w < target_w & ratio < target_ratio
            tf.logical_and(tf.less(ratio, target_ratio),
                           tf.logical_and(tf.greater_equal(new_w, tf.cast(tf.divide(target_w, 2), tf.int32)),
                                          tf.less(new_w, target_w))): pad_fn,
            # case 3 : new_w < target_w/2 & new_w < target_w & ratio < target_ratio
            tf.logical_and(tf.less(ratio, target_ratio),
                           tf.logical_and(tf.less(new_w, target_w),
                                          tf.less(new_w, tf.cast(tf.divide(target_w, 2), tf.int32)))): replicate_fn
        },
        default=simple_resize, exclusive=True)
    return pad_image, new_w  # new_w = image width used for computing sequence lengths