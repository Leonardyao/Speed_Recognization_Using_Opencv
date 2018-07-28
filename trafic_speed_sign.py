# -*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def img_deal(img, x, y, r):

    # cv2.IMREAD_COLOR
    # cv2.IMREAD_UNCHANGED
    # cv2.IMREAD_ANYDEPTH

    #img = cv2.imread(input_img)
    rows, cols, channel = img.shape

    # new image
    img_new = np.zeros((rows,cols,4),np.uint8)
    img_new[:,:,0:3] = img[:,:,0:3]

    # one channel pic
    img_circle = np.zeros((rows,cols,1),np.uint8)
    img_circle[:,:,:] = 0  # transparent
    img_circle = cv2.circle(img_circle,(x,y), r,(255), -1)

    # merge
    img_new[:,:,3] = img_circle[:,:,0]
    return img_new

def get_color(img):
    #print('get color')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # threshold to controll selection of red region
    maxsum = 0.4
    color = []
    color_dict = getColorList()
    w, h,_ = img.shape
    img_area = w * h
    for d in color_dict:
        mask = cv2.inRange(hsv, color_dict[d][0], color_dict[d][1])
        cv2.imwrite(d + '.jpg', mask)
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)
        img, cnts, hiera = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sum = 0
        for c in cnts:
            sum += cv2.contourArea(c)
        if sum/img_area > maxsum:
            color.append(d)
    return color


def getColorList():
    dict = {}

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    color_list = []
    color_list.append(lower_black)
    color_list.append(upper_black)
    dict['black'] = color_list

    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    color_list = []
    color_list.append(lower_white)
    color_list.append(upper_white)
    dict['white'] = color_list

    # 红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red'] = color_list

    # 红色2
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    color_list = []
    color_list.append(lower_red)
    color_list.append(upper_red)
    dict['red2'] = color_list

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    color_list = []
    color_list.append(lower_orange)
    color_list.append(upper_orange)
    dict['orange'] = color_list

    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    color_list = []
    color_list.append(lower_yellow)
    color_list.append(upper_yellow)
    dict['yellow'] = color_list

    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    color_list = []
    color_list.append(lower_green)
    color_list.append(upper_green)
    dict['green'] = color_list

    # 青色
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    color_list = []
    color_list.append(lower_cyan)
    color_list.append(upper_cyan)
    dict['cyan'] = color_list

    # 蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    color_list = []
    color_list.append(lower_blue)
    color_list.append(upper_blue)
    dict['blue'] = color_list

    # 紫色
    lower_purple = np.array([125, 43, 46])
    upper_purple = np.array([155, 255, 255])
    color_list = []
    color_list.append(lower_purple)
    color_list.append(upper_purple)
    dict['purple'] = color_list

    return dict

def calculate_distance(target, template):
    v1 = np.array(target)
    v2 = np.array(template)
    dist = np.sum(np.abs(v1-v2)) # L1
    #dist = np.sqrt(np.sum(np.square(v1 - v2))) # L2
    return dist

def template(img):
    # template
    lists = os.listdir('./template')
    w, h, _ = img.shape
    L2_distance = {}
    for template in lists:
        tpl = cv2.imread(str('./template/' + template))
        # print(str('./template/' + template))
        # cv2.imshow('tpl', tpl)
        # cv2.waitKey(0)
        min_distance = 1e7
        for i in range(1, w):
            tpl = cv2.resize(tpl, (i, i))
            # purpose image
            # cv2.imshow('template', tpl)
            # cv2.imshow('target', img)

            methods = [cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED]

            # w and h of template
            th, tw = tpl.shape[:2]
            for md in methods:
                result = cv2.matchTemplate(img, tpl, md)
                # location
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if md == cv2.TM_SQDIFF_NORMED:
                    tl = min_loc
                else:
                    tl = max_loc
                br = (tl[0] + tw, tl[1] + th)
                target = img[tl[0]:br[0], tl[1]:br[1]]
                distance = calculate_distance(target, tpl)
                if min_distance > distance:
                    min_distance = distance
                    # 绘targets
                    # cv2.rectangle(img, tl, br, (0, 0, 255), 2)
                    # cv2.imshow('match-' + np.str(md), img)
        L2_distance[template] = min_distance
    return L2_distance


if __name__ == '__main__':
    lists = os.listdir('./data')
    for name in lists:
        print(str('./data/' + name))
        img = cv2.imread(str('./data/' + name))
        #img = cv2.imread('./data/30.jpg')
        # blur filter
        img = cv2.medianBlur(img,5)
        cv2.imshow('blur', img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        a = .1
        b = .9
        min_data = 0
        max_data = 255
        # norm = a + ((img - min_data)*(b - a)/(max_data - min_data))
        cv2.imshow('blur', img)
        # cv2.imshow('norm', norm)
        cv2.waitKey(0)
        w, h, _ = img.shape
        # tuning minRadius and  maxRadius to adjust proper circle size of ROI region
        circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                    300, param1=100, param2=30, minRadius = 5, maxRadius = min(w, h))
        circles = circles1[0, :, :]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:

            if i[0] > i[2] and i[1] > i[2]:
                newimage = img[int(i[1] - i[2]):int(i[1] + i[2]), int(i[0] - i[2]):int(i[0] + i[2])]
                # cv2.imshow('newimage', newimage)
                # cv2.waitKey(0)
                if 'red' in get_color(newimage) or 'red2' in get_color(newimage):
                    img_roi = newimage
                    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 5)
                    draw = cv2.rectangle(img, (int(i[0] - i[2]), int(i[1] + i[2])), (int(i[0] + i[2]), int(i[1] - i[2])),
                                         (255, 255, 0), 5)

        print("圆心坐标", i[0], i[1])
        # cv2.imshow('result', img)
        cv2.imshow('image', img)
        cv2.imshow('result', img_roi)
        cv2.waitKey(0)
        distance = template(img_roi)
        print(distance)
        distance_sort = sorted(zip(distance.values(), distance.keys()))
        speed = distance_sort[0][1].split('.')[0]
        print('The sign is limit: ', speed)
