from tensorflow.keras.applications import VGG16
import tensorflow as tf
import numpy as np

model = VGG16(weights = 'imagenet')
vgg16_weights = model.get_weights()

weights = {
    'conv1_1' : tf.constant(vgg16_weights[0]),
    'conv1_2' : tf.constant(vgg16_weights[2]),

    'conv2_1' : tf.constant(vgg16_weights[4]),
    'conv2_2' : tf.constant(vgg16_weights[6]),

    'conv3_1' : tf.constant(vgg16_weights[8]),
    'conv3_2' : tf.constant(vgg16_weights[10]),
    'conv3_3' : tf.constant(vgg16_weights[12]),

    'conv4_1' : tf.constant(vgg16_weights[14]),
    'conv4_2' : tf.constant(vgg16_weights[16]),
    'conv4_3' : tf.constant(vgg16_weights[18]),

    'conv5_1' : tf.constant(vgg16_weights[20]),
    'conv5_2' : tf.constant(vgg16_weights[22]),
    'conv5_3' : tf.constant(vgg16_weights[24]),
}

biases = {
    'conv1_1' : tf.constant(vgg16_weights[1]),
    'conv1_2' : tf.constant(vgg16_weights[3]),

    'conv2_1' : tf.constant(vgg16_weights[5]),
    'conv2_2' : tf.constant(vgg16_weights[7]),

    'conv3_1' : tf.constant(vgg16_weights[9]),
    'conv3_2' : tf.constant(vgg16_weights[11]),
    'conv3_3' : tf.constant(vgg16_weights[13]),

    'conv4_1' : tf.constant(vgg16_weights[15]),
    'conv4_2' : tf.constant(vgg16_weights[17]),
    'conv4_3' : tf.constant(vgg16_weights[19]),

    'conv5_1' : tf.constant(vgg16_weights[21]),
    'conv5_2' : tf.constant(vgg16_weights[23]),
    'conv5_3' : tf.constant(vgg16_weights[25]),
}

# input layer and output layer
x = tf.placeholder(tf.float32, [None, 256, 256, 3])
y = tf.placeholder(tf.float32, [None, 256, 256, 21])

def fcn16(x):
  # for batch_normalization
  training_or_not = tf.placeholder(tf.bool)

  conv1_1 = tf.nn.conv2d(x,
                     weights['conv1_1'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv1_1 = tf.nn.relu(tf.add(conv1_1, biases['conv1_1']))
  conv1_2 = tf.nn.conv2d(conv1_1,
                     weights['conv1_2'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv1_2 = tf.nn.relu(tf.add(conv1_2, biases['conv1_2']))
  maxp1 = tf.nn.max_pool(conv1_2,
                       ksize = [1, 2, 2, 1],
                       strides = [1, 2, 2, 1],
                       padding = 'VALID')

  # Second convolution layers
  conv2_1 = tf.nn.conv2d(maxp1,
                     weights['conv2_1'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv2_1 = tf.nn.relu(tf.add(conv2_1, biases['conv2_1']))
  conv2_2 = tf.nn.conv2d(conv2_1,
                     weights['conv2_2'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv2_2= tf.nn.relu(tf.add(conv2_2, biases['conv2_2']))
  maxp2 = tf.nn.max_pool(conv2_2,
                       ksize = [1, 2, 2, 1],
                       strides = [1, 2, 2, 1],
                       padding = 'VALID')

  # third convolution layers
  conv3_1 = tf.nn.conv2d(maxp2,
                     weights['conv3_1'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv3_1 = tf.nn.relu(tf.add(conv3_1, biases['conv3_1']))
  conv3_2 = tf.nn.conv2d(conv3_1,
                     weights['conv3_2'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv3_2= tf.nn.relu(tf.add(conv3_2, biases['conv3_2']))
  conv3_3 = tf.nn.conv2d(conv3_2,
                     weights['conv3_3'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv3_3= tf.nn.relu(tf.add(conv3_3, biases['conv3_3']))
  maxp3 = tf.nn.max_pool(conv3_3,
                       ksize = [1, 2, 2, 1],
                       strides = [1, 2, 2, 1],
                       padding = 'VALID')

  # fourth convolution layers
  conv4_1 = tf.nn.conv2d(maxp3,
                     weights['conv4_1'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv4_1 = tf.nn.relu(tf.add(conv4_1, biases['conv4_1']))
  conv4_2 = tf.nn.conv2d(conv4_1,
                     weights['conv4_2'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv4_2= tf.nn.relu(tf.add(conv4_2, biases['conv4_2']))
  conv4_3 = tf.nn.conv2d(conv4_2,
                     weights['conv4_3'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv4_3= tf.nn.relu(tf.add(conv4_3, biases['conv4_3']))
  maxp4 = tf.nn.max_pool(conv4_3,
                       ksize = [1, 2, 2, 1],
                       strides = [1, 2, 2, 1],
                       padding = 'VALID')

  # fifth convolution layers
  conv5_1 = tf.nn.conv2d(maxp4,
                     weights['conv5_1'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv5_1 = tf.nn.relu(tf.add(conv5_1, biases['conv5_1']))
  conv5_2 = tf.nn.conv2d(conv5_1,
                     weights['conv5_2'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv5_2= tf.nn.relu(tf.add(conv5_2, biases['conv5_2']))
  conv5_3 = tf.nn.conv2d(conv5_2,
                     weights['conv5_3'],
                     strides = [1, 1, 1, 1],
                     padding = 'SAME')
  conv5_3= tf.nn.relu(tf.add(conv5_3, biases['conv5_3']))

  maxp5 = tf.nn.max_pool(conv5_3,
                       ksize = [1, 2, 2, 1],
                       strides = [1, 2, 2, 1],
                       padding = 'VALID')

#output = tf.layers.conv2d_transpose(maxp5,
#                                  filters = 32,
#                                  kernel_size = 8,
#                                  strides = 32,
#                                  activation = tf.nn.relu,
#                                  padding = 'SAME')
#output = tf.layers.conv2d(output,
#                        filters = 21,
#                        kernel_size = 1,
#                        padding = 'SAME')
#
  maxp5_2x = tf.layers.conv2d_transpose(maxp5,
                                  filters = 512,
                                  kernel_size = 3,
                                  strides = (2, 2),
                                  activation = tf.nn.relu,
                                  padding = 'SAME')

  score1 = tf.layers.batch_normalization(maxp5_2x + maxp4, training=training_or_not)

  score2 = tf.layers.conv2d_transpose(score1,
                                  filters = 256,
                                  kernel_size = 3,
                                  strides = (2, 2),
                                  activation = tf.nn.relu,
                                  padding = 'SAME')

  score2 = tf.layers.batch_normalization(score2, training=training_or_not)

  score3 = tf.layers.conv2d_transpose(score2,
                                  filters = 128,
                                  kernel_size = 3,
                                  strides = (2, 2),
                                  activation = tf.nn.relu,
                                  padding = 'SAME')

  score3 = tf.layers.batch_normalization(score3, training=training_or_not)
  score4 = tf.layers.conv2d_transpose(score3,
                                  filters = 64,
                                  kernel_size = 3,
                                  strides = (2, 2),
                                  activation = tf.nn.relu,
                                  padding = 'SAME')
  score4 = tf.layers.batch_normalization(score4, training=training_or_not)

  score5 = tf.layers.conv2d_transpose(score4,
                                  filters = 32,
                                  kernel_size = 3,
                                  strides = (2, 2),
                                  activation = tf.nn.relu,
                                  padding = 'SAME')
  score5 = tf.layers.batch_normalization(score5, training=training_or_not)

  output = tf.layers.conv2d(score5,
                        filters = 21,
                        kernel_size = 1,
                        padding = 'SAME')
  return output
