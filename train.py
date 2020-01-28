import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
import tensorflow as tf
from PIL import Image
from os.path import join, splitext, abspath
from os import listdir
import os
import time
from tqdm import tqdm

tf.device("/gpu:0")

data_path = 'C:\\Users\\chowy\\Downloads\\VOCtrainval_11-May-2012'
main_path = join(data_path, 'VOCdevkit', 'VOC2012')
img_path = join(main_path, 'JPEG')
ann_path = join(main_path, 'SegmentationClass')

img_list = listdir(img_path)
ann_list = listdir(ann_path)

img_list.sort()
ann_list.sort()

n_classes = 21

def get_pascal_labels():
    return np.asarray([
                        [0, 0, 0],
                        [128, 0, 0],
                        [0, 128, 0],
                        [128, 128, 0],
                        [0, 0, 128],
                        [128, 0, 128],
                        [0, 128, 128],
                        [128, 128, 128],
                        [64, 0, 0],
                        [192, 0, 0],
                        [64, 128, 0],
                        [192, 128, 0],
                        [64, 0, 128],
                        [192, 0, 128],
                        [64, 128, 128],
                        [192, 128, 128],
                        [0, 64, 0],
                        [128, 64, 0],
                        [0, 192, 0],
                        [128, 192, 0],
                        [0, 64, 128],
            ])

# 이미지 2912까지있음
training_data=[]
training_seg=[]
test_data = []
training_data = np.load('./data/training_data.npy')
training_seg = np.load('./data/training_seg.npy')

training_seg.shape

## data channel change 3=>21
#for index in tqdm(range(500)):
#    img = Image.open(join(img_path, splitext(ann_list[index])[0]+'.jpg'))
#    ann = Image.open(join(ann_path, ann_list[index])).convert('RGB')
#    img = img.resize((576,160))
#    ann = ann.resize((576,160))
#    img = np.array(img)/255
#    ann = np.array(ann)
#    print(ann.shape)
#    break
#    depth_21 =[]
#    training_data.append(img)
#    for ii , k in enumerate(get_pascal_labels()):
#        cat_channel = np.zeros((ann.shape[0], ann.shape[1]), dtype=np.int16)
#        for i in range(ann.shape[0]):
#            for j in range(ann.shape[1]):
#                if np.all(ann[i][j] == k, axis=-1):
#                    cat_channel[i][j] = 1
#        depth_21.append(cat_channel)
#    training_seg.append(depth_21)
#np.save('C:\\Users\\chowy\\Desktop\\workspace\\data\\training_data',training_data)
#np.save('C:\\Users\\chowy\\Desktop\\workspace\\data\\training_seg',training_seg)


# make test set
for index in range(1100, 1200):
    img = Image.open(join(img_path, splitext(ann_list[index])[0]+'.jpg'))
    img = img.resize((256,256))
    img = np.array(img)/255
    test_data.append(img)

    if index == 100:
        print('only half!')
    else: pass


#seg_GT_check
#seg_check_index = 1
#rgb1 = np.zeros((training_seg[seg_check_index].shape[0],training_seg[seg_check_index].shape[1], 3), dtype=np.uint8)
#rgb1[:][:] = [224, 224, 192]
#for n in range(160):
#    for m in range(576):
#         for b, c in enumerate(get_pascal_labels()):
#            #len(np.where(test_seg[n][m]>0.4)[0])
#            if training_seg[seg_check_index][n][m][b] == 1:
#                 rgb1[n, m, 0] = c[0]
#                 rgb1[n, m, 1] = c[1]
#                 rgb1[n, m, 2] = c[2]
#plt.imshow(rgb1)

print('training_data_shape = ', np.shape(training_data))
print('training_seg_shape = ', np.shape(training_seg))

print('Data ready!')

#모델
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

score2 = tf.layers.batch_normalization(score2 + maxp3, training=training_or_not)

score3 = tf.layers.conv2d_transpose(score2,
                                  filters = 32,
                                  kernel_size = 12,
                                  strides = 8,
                                  activation = tf.nn.relu,
                                  padding = 'SAME')
score3
#score3 = tf.layers.conv2d_transpose(score2,
#                                  filters = 128,
#                                  kernel_size = 3,
#                                  strides = (2, 2),
#                                  activation = tf.nn.relu,
#                                  padding = 'SAME')
#score3
#score3 = tf.layers.batch_normalization(score3, training=training_or_not)
#score4 = tf.layers.conv2d_transpose(score3,
#                                  filters = 64,
#                                  kernel_size = 3,
#                                  strides = (2, 2),
#                                  activation = tf.nn.relu,
#                                  padding = 'SAME')
#score4 = tf.layers.batch_normalization(score4, training=training_or_not)
#
#score5 = tf.layers.conv2d_transpose(score4,
#                                  filters = 32,
#                                  kernel_size = 3,
#                                  strides = (2, 2),
#                                  activation = tf.nn.relu,
#                                  padding = 'SAME')
score5 = tf.layers.batch_normalization(score3, training=training_or_not)

output = tf.layers.conv2d(score5,
                        filters = 21,
                        kernel_size = 1,
                        padding = 'SAME')

## 1x1 convolution layers
#fcn4 = tf.layers.conv2d(conv6,
#                        filters = 4096,
#                        kernel_size = 1,
#                        padding = 'SAME',
#                        activation = tf.nn.relu)
output
# 컴파일
LR  = 0.0001 # or 0.00001 fix

pred = output

logits = tf.reshape(pred, (-1,21))
labels = tf.reshape(y, (-1,21))
loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
loss = tf.reduce_mean(loss)
update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS)
optm  = tf.train.AdamOptimizer(LR).minimize(loss)
optm = tf.group([optm, update_ops])



n_batch = 64
n_epoch = 3500
n_prt = 100
n_train= 1000

def train_batch_maker(batch_size) :
    n_train_batch=[]
    n_train_seg_batch=[]
    for i in range(batch_size):
        random_idx = np.random.randint(1000)
        n_train_batch.append(training_data[random_idx])
        n_train_seg_batch.append(training_seg[random_idx])
    return n_train_batch, n_train_seg_batch


sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_record_train = []

for epoch in range(n_epoch):
    ran_num = np.random.randint(n_train)
    train_x, train_y = train_batch_maker(n_batch) #(n, 160, 576, 3)
    sess.run(optm, feed_dict = {x: train_x, y: train_y, training_or_not: True})

    if epoch % n_prt == 0:
        c = sess.run(loss, feed_dict = {x: train_x, y: train_y, training_or_not: False})
        loss_record_train.append(c)
        print ("Epoch : {}".format(epoch))
        print ("Cost : {}".format(c))

#loss graph
#plt.fure(figsize = (10,8))
#plt.plot(np.arange(len(loss_record_train))*n_prt, loss_record_train, label = 'training')
#plt.xlabel('epoch', fontsize = 15)
#plt.ylabel('loss', fontsize = 15)
#plt.legend(fontsize = 12)
#plt.ylim([0, np.max(loss_record_train)])
#plt.show()

### test
#test_img = sess.run(tf.nn.softmax(logits), feed_dict = {x: test_data, training_or_not:False})
#
#test_img = test_img.reshape(-1,256,256,21)
#
#x = np.argmax(test_img, axis=-1)
#
#for seg_check_index in range(100):
#
#    rgb1 = np.zeros((x[seg_check_index].shape[0],test_img[seg_check_index].shape[1], 3), dtype=np.uint8)
#    #rgb1 = np.zeros((x[seg_check_index].shape[0],test_img[seg_check_index].shape[1], 3), dtype=np.float32)
#    rgb1[:][:] = [224, 224, 192]
#
#    for n in range(256):
#        for m in range(256):
#            for b, c in enumerate(get_pascal_labels()):
#            #len(np.where(test_seg[n][m]>0.4)[0])
#                if x[seg_check_index][n][m] == b:
#                    rgb1[n, m, 0] = c[0]
#                    rgb1[n, m, 1] = c[1]
#                    rgb1[n, m, 2] = c[2]
#
#    plt.imshow(rgb1)
#    plt.savefig('./result/test{}.png'.format(seg_check_index))
#
