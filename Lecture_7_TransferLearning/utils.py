import os
import numpy as np
from vgg_preprocess import preprocess_for_train
import tensorflow as tf

def get_file(file_dir):
    labels = []
    images = []
    for root, _, files in os.walk(file_dir):
        for name in files:
            cls_name = name.split('.')[0]
            if cls_name == 'cat':
                labels.append(0)
                images.append(file_dir + name)
            else:
                labels.append(1)
                images.append(file_dir + name)

    temp = np.array([images, labels])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(float(i)) for i in label_list]
    
    return image_list, label_list

 
img_width = 224
img_height = 224

def get_batch(image_list, label_list, img_width, img_height, batch_size, capacity):
    image = tf.cast(image_list, tf.string)
    label = tf.cast(label_list, tf.int32)
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])

    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = preprocess_for_train(image, 224, 224)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch, label_batch