__author__ = 'solivr'
__license__ = "GPL"


import tensorflow as tf
import requests
from typing import Union, List
from .config import Params, CONST
from .data_utils import augment_data, padding_inputs_width


def data_loader(csv_filename: Union[List[str], str], params: Params, labels=True, batch_size: int=64,
                data_augmentation: bool=True, num_epochs: int=None, image_summaries: bool=True):
    """
    Loads, preprocesses (data augmentation, padding) and feeds the data
    :param csv_filename: filename or list of filenames
    :param params: Params object containing all the parameters
    :param labels: transcription labels
    :param batch_size: batch_size
    :param data_augmentation: flag to select or not data augmentation
    :param num_epochs: feeds the data 'num_epochs' times
    :param image_summaries: flag to show image summaries or not
    :return: data_loader function
    """

    padding = True

    def input_fn():
        if labels:
            csv_types = [['None'], ['None'], ['None'], ['None'], ['None'], ['None'], ['None'], ['None']]
        else:
            csv_types = [['None']]

        dataset = tf.data.experimental.CsvDataset(csv_filename, record_defaults=csv_types, header=False, field_delim=params.csv_delimiter, use_quote_delim=True)

        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1024, count=num_epochs))


        # -- Read image
        def _image_reading_preprocessing(path, width, height, label, xmin, ymin, xmax, ymax) -> dict():

            # Load image from url
            # image_content = requests.get(img_url).content

            # Load image from path
            image_content = tf.read_file(path, name='filename_reader')
            # decoding image
            image = tf.cond(
                tf.image.is_jpeg(image_content),
                lambda: tf.image.decode_jpeg(image_content, channels=params.input_channels, name="image_decoding_op", try_recover_truncated=True),
                lambda: tf.image.decode_png(image_content, channels=params.input_channels, name='image_decoding_op'))

            # Image Cropping
            # clip by value
            # augmentation (box size) => exception
            # xmin, xmax, ymin, ymax에 대해 clip by value로 exception 피하기

            xmin = tf.string_to_number(xmin, out_type=tf.float32)
            ymin = tf.string_to_number(ymin, out_type=tf.float32)
            xmax = tf.string_to_number(xmax, out_type=tf.float32)
            ymax = tf.string_to_number(ymax, out_type=tf.float32)
            height = tf.string_to_number(height, out_type=tf.float32)
            width = tf.string_to_number(width, out_type=tf.float32)

            x_ratio = tf.random_uniform([], minval=-0.05, maxval=0.05, dtype=tf.float32)
            y_ratio = tf.random_uniform([], minval=-0.05, maxval=0.05, dtype=tf.float32)

            box_width = tf.subtract(xmax, xmin)
            box_height = tf.subtract(ymax, ymin)
            xmin = tf.clip_by_value(tf.add(xmin, tf.multiply(box_width, x_ratio)), 0, width)
            ymin = tf.clip_by_value(tf.add(ymin, tf.multiply(box_height, y_ratio)), 0, height)
            xmax = tf.clip_by_value(tf.add(xmax, tf.multiply(box_width, x_ratio)), 0, width)
            ymax = tf.clip_by_value(tf.add(ymax, tf.multiply(box_height, x_ratio)), 0, height)
            image = tf.image.crop_to_bounding_box(image, tf.cast(ymin, tf.int32), tf.cast(xmin, tf.int32),
                                                  tf.cast(tf.subtract(ymax, ymin), tf.int32),
                                                  tf.cast(tf.subtract(xmax, xmin), tf.int32))

            # Data augmentation
            if data_augmentation:
                image = augment_data(image, params.data_augmentation_max_rotation)

            # Padding
            if padding:
                with tf.name_scope('padding'):
                    image, img_width = padding_inputs_width(image, target_shape=params.input_shape,
                                                            increment=CONST.DIMENSION_REDUCTION_W_POOLING)
            # Resize
            else:
                image = tf.image.resize_images(image, size=params.input_shape)
                img_width = tf.shape(image)[1] # image shape : [H, W, C]

            # Update features
            features = {'filenames': path, 'labels': label}
            # features = {'image_url': img_url, 'labels': label} # image_url features
            features.update({'images': image, 'images_widths': img_width})

            return features

        dataset = dataset.map(_image_reading_preprocessing, num_parallel_calls=params.input_data_n_parallel_calls)

        dataset = dataset.batch(batch_size).prefetch(32)
        prepared_batch = dataset.make_one_shot_iterator().get_next()

        if image_summaries:
            tf.summary.image('input/image', prepared_batch['images'], max_outputs=1)
        if labels:
            tf.summary.text('input/labels', prepared_batch.get('labels')[:10])

        return prepared_batch, prepared_batch.get('labels')

    return input_fn


def serving_single_input(params : Params, fixed_height: int=32, min_width: int=8):
    """
    Serving input function needed for export (in TensorFlow).
    Features to serve :
        - `images` : greyscale image
        - `input_filename` : filename of image segment
        - `input_url` : image url of image segment
        - `input_rgb`: RGB image segment
    :param fixed_height: height  of the image to format the input data with
    :param min_width: minimum width to resize the image
    :param params : Params
    :return: serving_input_fn
    """

    def serving_input_fn():

        # define placeholder for image url, xmin, ymin, xmax, ymax, height, width
        filename = tf.placeholder(dtype=tf.string)
        # img_url = tf.placeholder(dtype=tf.string)
        xmin = tf.placeholder(dtype=tf.float32)
        ymin = tf.placeholder(dtype=tf.float32)
        xmax = tf.placeholder(dtype=tf.float32)
        ymax = tf.placeholder(dtype=tf.float32)
        height = tf.placeholder(dtype=tf.float32)
        width = tf.placeholder(dtype=tf.float32)

        x_ratio = tf.random_uniform([], minval=-0.05, maxval=0.05, dtype=tf.float32)
        y_ratio = tf.random_uniform([], minval=-0.05, maxval=0.05, dtype=tf.float32)

        box_width = tf.subtract(xmax, xmin)
        box_height = tf.subtract(ymax, ymin)
        xmin = tf.clip_by_value(tf.add(xmin, tf.multiply(box_width, x_ratio)), 0, width)
        ymin = tf.clip_by_value(tf.add(ymin, tf.multiply(box_height, y_ratio)), 0, height)
        xmax = tf.clip_by_value(tf.add(xmax, tf.multiply(box_width, x_ratio)), 0, width)
        ymax = tf.clip_by_value(tf.add(ymax, tf.multiply(box_height, x_ratio)), 0, height)

        # image_content = requests.get(img_url).content
        image_content = tf.read_file(filename, name='filename_reader')
        decoded_image = tf.image.decode_jpeg(image_content, channels=params.input_channels, name="image_decoding_op",
                             try_recover_truncated=True)

        image = tf.to_float(tf.image.crop_to_bounding_box(decoded_image, tf.cast(ymin, tf.int32), tf.cast(xmin, tf.int32),
                                              tf.cast(tf.subtract(ymax, ymin), tf.int32),
                                              tf.cast(tf.subtract(xmax, xmin), tf.int32)))

        image = tf.image.rgb_to_grayscale(image, name='rgb2gray')

        shape = tf.shape(image)

        # Assert image is gray-scale
        # assert shape[2] == 1

        ratio = tf.divide(shape[1], shape[0])
        increment = CONST.DIMENSION_REDUCTION_W_POOLING
        new_width = tf.cast(tf.round((ratio * fixed_height) / increment) * increment, tf.int32)

        resized_image = tf.cond(new_width < tf.constant(min_width, dtype=tf.int32),
                                true_fn=lambda: tf.image.resize_images(image, size=(fixed_height, min_width)),
                                false_fn=lambda: tf.image.resize_images(image, size=(fixed_height, new_width))
                                )

        # Features to serve
        features = {'images': resized_image[None],  # cast to 1 x h x w x c
                    'images_widths': new_width[None]  # cast to tensor
                    }

        # Inputs received
        receiver_inputs = {'images': image}
        alternative_receivers = {'input_filename': {'filename': filename}, 'input_rgb': {'rgb_images': decoded_image}}
        # alternative_receivers = {'input_img_url': {'img_url': img_url}, 'input_rgb': {'rgb_images': decoded_image}}

        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors=receiver_inputs,
                                                        receiver_tensors_alternatives=alternative_receivers)

    return serving_input_fn
