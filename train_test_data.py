import os
import random

train_data = []
test_data = []
abs_path = os.path.dirname(os.path.abspath(__file__))
images_path = os.path.join(abs_path,  'images')
labels_path = os.path.join(abs_path,  'labels')

data_num = 500

''' generate text-line dataset '''
for idx in range(data_num ):
    image_tag = os.path.join(images_path, f'image_{(idx+1)}.jpg')
    with open(os.path.join(labels_path, f'label_{(idx+1)}.txt'), "r", encoding= 'utf-8') as file:
        label_tag = file.read()
    if idx % 5 != 1:
        train_data.append(image_tag+'\t'+label_tag)
    else :
        test_data.append(image_tag+'\t'+label_tag)

''' write the data into .txt file '''
train_path = os.path.join(abs_path, 'train.txt')
test_path = os.path.join(abs_path, 'test.txt')

with open(train_path, "w", encoding='utf-8') as file:
    for item in train_data:
        file.write(f'{item}'+'\n')

with open(test_path, "w", encoding= 'utf-8') as file:
    for item in test_data:
        file.write(rf'{item}'+'\n')