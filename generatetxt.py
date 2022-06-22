import os

trian_path = 'data/custom/images/train'
valid_path = 'data/custom/images/valid'
trian_txt_path = 'data/custom/train.txt'
valid_txt_path = 'data/custom/valid.txt'

train_dir = os.listdir(trian_path)
valid_dir = os.listdir(trian_path)

with open(trian_txt_path,'w+') as f:
    for image_name in train_dir:
        f.write(trian_path + '/' + image_name +'\n')

with open(valid_txt_path,'w+') as f:
    for image_name in valid_dir:
        f.write(valid_path + '/' + image_name +'\n')
print('index has generated')
