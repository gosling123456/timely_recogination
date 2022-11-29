# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result, allFilePath, init_model, cv_imread
# from plate_recognition.plate_cls import cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from plate_recognition.color_rec import plate_color_rec, init_color_model

clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    # clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords


def get_plate_rec_landmark(img, xyxy, conf, landmarks, class_num, device, plate_rec_model, plate_color_model=None):
    h, w, c = img.shape
    result_dict = {}
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness

    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    height = y2 - y1
    landmarks_np = np.zeros((4, 2))
    rect = [x1, y1, x2, y2]
    for i in range(4):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        landmarks_np[i] = np.array([point_x, point_y])

    class_label = int(class_num)  # 车牌的的类型0代表单牌，1代表双层车牌
    roi_img = four_point_transform(img, landmarks_np)  # 透视变换得到车牌小图
    color_code = plate_color_rec(roi_img, plate_color_model, device)  # 车牌颜色识别
    if class_label:  # 判断是否是双层车牌，是双牌的话进行分割后然后拼接
        roi_img = get_split_merge(roi_img)
    plate_number = get_plate_result(roi_img, device, plate_rec_model)  # 对车牌小图进行识别
    # cv2.imwrite("roi.jpg",roi_img)
    result_dict['rect'] = rect  # 车牌roi区域
    result_dict['landmarks'] = landmarks_np.tolist()  # 车牌角点坐标
    result_dict['plate_no'] = plate_number  # 车牌号
    result_dict['roi_height'] = roi_img.shape[0]  # 车牌高度
    result_dict['plate_color'] = color_code  # 车牌颜色
    result_dict['plate_type'] = class_label  # 单双层 0单层 1双层
    return result_dict


def detect_Recognition_plate(model, orgimg, device, plate_rec_model, img_size, plate_color_model=None):
    conf_thres = 0.3
    iou_thres = 0.5
    dict_list = []
    # orgimg = cv2.imread(image_path)  # BGR
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found '
    h0, w0 = orgimg.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size
    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    # Run inference
    t0 = time.time()

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = model(img)[0]
    t2 = time_synchronized()
    # print(f"infer time is {(t2-t1)*1000} ms")

    # Apply NMS
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

            for j in range(det.size()[0]):
                xyxy = det[j, :4].view(-1).tolist()
                conf = det[j, 4].cpu().numpy()
                landmarks = det[j, 5:13].view(-1).tolist()
                class_num = det[j, 13].cpu().numpy()
                result_dict = get_plate_rec_landmark(orgimg, xyxy, conf, landmarks, class_num, device, plate_rec_model,
                                                     plate_color_model)
                dict_list.append(result_dict)
    return dict_list


def draw_result(orgimg, dict_list):
    result_str = ""
    for result in dict_list:
        rect_area = result['rect']

        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = max(0, int(y - padding_h))
        rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        height_area = result['roi_height']
        landmarks = result['landmarks']
        result_p = result['plate_no']
        if result['plate_type'] == 0:
            result_p += " " + result['plate_color']
        else:
            result_p += " " + result['plate_color'] + "双层"
        result_str += result_p + " "
        for i in range(4):  # 关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)  # 画框
        if len(result) >= 1:
            orgimg = cv2ImgAddText(orgimg, result_p, rect_area[0] - height_area, rect_area[1] - height_area - 10,
                                   (0, 255, 0), height_area)
    print(result_str)
    return orgimg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/plate_detect.pt',
                        help='model.pt path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec.pth', help='model.pt path(s)')  # 识别模型
    parser.add_argument('--color_model', type=str, default='weights/color_classify.pth', help='plate color')  # 颜色识别模型
    parser.add_argument('--source', type=str, default='0', help='source')
    # parser.add_argument('--image_path', type=str, default='imgs', help='source')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='result1', help='source')
    parser.add_argument('--video', type=str, default='', help='source')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device =torch.device("cpu")
    opt = parser.parse_args()
    print(opt)
    save_path = opt.output
    count = 0
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    detect_model = load_model(opt.detect_model, device)  # 初始化检测模型
    plate_rec_model = init_model(device, opt.rec_model)  # 初始化识别模型
    # 算参数量
    total = sum(p.numel() for p in detect_model.parameters())
    total_1 = sum(p.numel() for p in plate_rec_model.parameters())
    print("detect params: %.2fM,rec params: %.2fM" % (total / 1e6, total_1 / 1e6))

    plate_color_model = init_color_model(opt.color_model, device)
    time_all = 0
    time_begin = time.time()
    print(opt.source,type(opt.source))

    if opt.source == '0':  # 调用摄像头
        path = 'videos/%s.avi' % '_'.join(time.ctime(time.time()).split(' ')).replace(':', '_')  # 以当前时刻命名
        video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 5, (1280, 720))  # mp4v 才能生成MP4格式的文件，其他会报错
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()
        while (True):
            ref, img = cap.read()
            if img is None:
                continue
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # detect_one(model,img_path,device)
            dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                 plate_color_model)
            ori_img = draw_result(img, dict_list)
            video.write(ori_img)
            cv2.imshow('1', ori_img)
            c = cv2.waitKey(100) & 0xff
            if c == 27:
                cap.release()
                break
        cv2.destroyAllWindows()

    elif opt.source.split('.')[-1] in ['jpg','png','jfif']:
        img = cv2.imread(opt.source)
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        # detect_one(model,img_path,device)
        dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                             plate_color_model)
        ori_img = draw_result(img, dict_list)
        cv2.imwrite('resule1'+'\\%s' % opt.source.split('\\')[-1], ori_img)
        cv2.imshow('1', ori_img)
        cv2.waitKey(1000)
        print('结果已成功保存至：','%s'%opt.source.split('\\')[-1])
        cv2.destroyAllWindows()


    elif opt.source.split('.')[-1].lower() in ['mp4']:
        print('终于到我了')
        vidcap = cv2.VideoCapture(opt.source)
        success, image = vidcap.read()
        count = 0
        path = 'videos/%s.avi' % '_'.join(time.ctime(time.time()).split(' ')).replace(':', '_')  # 以当前时刻命名
        video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 5, (1280, 720))  # mp4v 才能生成MP4格式的文件，其他会报错
        while success:
            success, img = vidcap.read()
            if img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # detect_one(model,img_path,device)
            dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                 plate_color_model)
            ori_img = draw_result(img, dict_list)
            video.write(ori_img)
            # cv2.imwrite('resule1' + '\\%s' % opt.source.split('\\')[-1], ori_img)
            cv2.imshow('1', ori_img)
            c = cv2.waitKey(1) & 0xff
            if c == 27:
                break
            print('播放')
        cv2.destroyAllWindows()
        video.release()

    elif os.path.isdir(opt.source):
        path = 'videos/%s.avi' % '_'.join(time.ctime(time.time()).split(' ')).replace(':', '_')# 以当前时刻命名
        video = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'MJPG'), 5, (1280, 720))  # mp4v 才能生成MP4格式的文件，其他会报错
        for i in os.listdir(opt.source):
            print(opt.source+'\\'+i)
            if i.split('.')[-1] in ['jpg']:
                img = cv2.imread(opt.source+'\\'+i)
                if img.shape[-1] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                # detect_one(model,img_path,device)
                dict_list = detect_Recognition_plate(detect_model, img, device, plate_rec_model, opt.img_size,
                                                     plate_color_model)
                ori_img = draw_result(img, dict_list)
                # cv2.imwrite('resule1' + '\\%s' % opt.source.split('\\')[-1], ori_img)
                video.write(ori_img)
                cv2.imshow('1', ori_img)
                cv2.waitKey(1000)
                c = cv2.waitKey(100) & 0xff
                if c == 27:
                    break
        cv2.destroyAllWindows()
        video.release()
        print('已保存！')

