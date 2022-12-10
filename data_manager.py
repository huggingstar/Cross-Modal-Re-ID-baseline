from __future__ import print_function, absolute_import
import os
import numpy as np
import random


def get_cam_id_from_full_path(image_path):
    '''根据 image path 获取 camera id'''
    path_split_list = image_path.split(os.path.sep)
    cam_name = path_split_list[-3]
    cam_id = cam_name[-1]
    return cam_id


def get_person_id_from_full_path(image_path):
    '''根据 image path 获取 person id'''
    path_split_list = image_path.split(os.path.sep)
    person_id = path_split_list[-2]
    return person_id


def get_id_list_from_txt_file(file_path):
    '''list 中的 id 都是4个字符的 string'''
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]


def process_query_sysu(data_path, mode='all', relabel=False):
    '''test_id.txt 中的id在所有ir镜头下的照片'''
    if mode == 'all':
        ir_cameras = ['cam3', 'cam6']
    elif mode == 'indoor':
        ir_cameras = ['cam3', 'cam6']

    # file_path = os.path.join(data_path, 'exp/test_id.txt')
    file_path = os.path.join(data_path, 'exp', 'test_id.txt')
    files_rgb = []
    files_ir = []

    # with open(file_path, 'r') as file:
    #     ids = file.read().splitlines()
    #     ids = [int(y) for y in ids[0].split(',')]
    #     ids = ["%04d" % x for x in ids]
    # add
    ids = get_id_list_from_txt_file(file_path)

    for id in sorted(ids):
        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                # new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                # 把同一个 id 在所有 ir_camera 中的图片地址保存到一个 list 中
                new_files_path = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                # 把所有 ir 图片的地址合并到一个 list 中
                # 一维 list
                files_ir.extend(new_files_path)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        # pid: person id
        # camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        # add
        cam_id = get_cam_id_from_full_path(img_path)
        pid = get_person_id_from_full_path(img_path)
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(cam_id)
    return query_img, np.array(query_id), np.array(query_cam)


def process_gallery_sysu(data_path, mode='all', trial=0, relabel=False):
    '''test_id.txt 中的id在每个 rgb 摄像头下随机抽取的一张照片'''
    random.seed(trial)

    if mode == 'all':
        rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
    elif mode == 'indoor':
        rgb_cameras = ['cam1', 'cam2']

    # file_path = os.path.join(data_path, 'exp/test_id.txt')
    file_path = os.path.join(data_path, 'exp', 'test_id.txt')
    files_rgb = []
    # with open(file_path, 'r') as file:
    #     ids = file.read().splitlines()
    #     ids = [int(y) for y in ids[0].split(',')]
    #     ids = ["%04d" % x for x in ids]
    # add
    ids = get_id_list_from_txt_file(file_path)

    # 每个 pid 在每个 rgb camera 下只抽取一张照片
    for id in sorted(ids):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                # new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                # 随机抽取一张
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        # camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        cam_id = get_cam_id_from_full_path(img_path)
        pid = get_person_id_from_full_path(img_path)
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(cam_id)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, trial=1, modal='visible'):
    if modal == 'visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 'thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, np.array(file_label)
