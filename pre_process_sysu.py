import numpy as np
from PIL import Image
import pdb
import os
import local_path

# 预处理干了什么？

# data_path = '/home/datasets/prml/computervision/re-id/sysu-mm01/ori_data'
data_path = local_path.my_test_SYSU_MM01
# my_test_note.createSmallTestSet(retain_dir_num=30, retain_file_num=5)


rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

# load id info
# file_path_train = os.path.join(data_path,'exp/train_id.txt')
# file_path_val   = os.path.join(data_path,'exp/val_id.txt')


# file_path_train = os.path.join(data_path, 'exp\\train_id.txt')
# file_path_val = os.path.join(data_path, 'exp\\val_id.txt')
file_path_train = os.path.join(data_path, 'exp', 'train_id.txt')
file_path_val = os.path.join(data_path, 'exp', 'val_id.txt')

# id_train: id in train set
with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    # 设置字符串格式， 至少4位，不足4位，在左侧用0补足
    id_train = ["%04d" % x for x in ids]

# id_val: id in validation set
with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]

# 合并 train_id.txt 和 val_id.txt 中的 pe rson id
# combine train and val split
# 获取所有的 id for train
id_train.extend(id_val)

# 分别提取 train set 中所有 rgb 和 ir 图片的文件路径
# 保存 train set 中所有 rgb 图片的文件路径
files_rgb = []  # datatype：str
# 保存 train set 中所有 ir 图片的文件路径
files_ir = []  # datatype：str
for id in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            # polish
            # new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)

    for cam in ir_cameras:
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            # polish
            # new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
            new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
            files_ir.extend(new_files)


# add
def getPid(img_path):
    '''从完整的图片路径中提取 person id'''
    path_split_list = img_path.split(sep=os.path.sep)
    # 倒数第二个元素为 person id
    pid = int(path_split_list[-2])
    return pid


# relabel
# pid_container：所有ir照片都在哪些文件夹中
pid_container = set()
for img_path in files_ir:
    # pid: person id
    # pid = int(img_path[-13:-9])
    # add
    pid = getPid(img_path)
    pid_container.add(pid)
pid2label = {pid: label for label, pid in enumerate(pid_container)}

# 把所有 ir 图片的 person id 放到 set 中
# ir 图片的 person id 和汇总后的 train_id 一样吗？ 在构造的小数据集中是一样的（当前小数据集的构造方式有问题）
# 验证代码
# id_train_set = set([int(i) for i in id_train])
# print(id_train_set == pid_container)



# 统一图片的分辨率
fix_image_width = 144
fix_image_height = 288


def read_imgs(train_image):
    # 以 ndarray 保存所有图片
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        # img.size    # W × H  宽 × 高
        # img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        img = img.resize((fix_image_width, fix_image_height), Image.Resampling.LANCZOS)
        pix_array = np.array(img)
        # pix_array.shape # H × W × C 高 × 宽 × 通道数

        train_img.append(pix_array)

        # label
        # pid = int(img_path[-13:-9])
        pid = getPid(img_path)
        # relabel
        # pid = pid2label[pid]
        label = pid2label[pid]
        train_label.append(label)
    # 把所有图片包装到一个 ndarray 中， 把所有 label 包装到一个 ndarray 中
    return np.array(train_img), np.array(train_label)


# rgb imges
train_img, train_label = read_imgs(files_rgb)
np.save(data_path + 'train_rgb_resized_img.npy', train_img)
np.save(data_path + 'train_rgb_resized_label.npy', train_label)

# ir imges
train_img, train_label = read_imgs(files_ir)
np.save(data_path + 'train_ir_resized_img.npy', train_img)
np.save(data_path + 'train_ir_resized_label.npy', train_label)
print('############ pre_process_sysu done. ############')
