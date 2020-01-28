import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os.path import join, splitext, abspath
from os import listdir
import os
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

