import argparse

import numpy as np
import sys
import imageio
import os
import torch
import math
from tqdm import tqdm
from skimage.transform import resize

import scipy
import scipy.io
import scipy.misc

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale

from skimage.feature import match_descriptors
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform

import pylab as pl

import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2


#鼠标事件
def get_rect(im, title='get_rect'):   #   (a,b) = get_rect(im, title='get_rect')
    mouse_params = {'tl': None, 'br': None, 'current_pos': None,
        'released_once': False}

    cv2.namedWindow(title)
    cv2.moveWindow(title, 100, 100)

    def onMouse(event, x, y, flags, param):

        param['current_pos'] = (x, y)

        if param['tl'] is not None and not (flags & cv2.EVENT_FLAG_LBUTTON):
            param['released_once'] = True

        if flags & cv2.EVENT_FLAG_LBUTTON:
            if param['tl'] is None:
                param['tl'] = param['current_pos']
            elif param['released_once']:
                param['br'] = param['current_pos']


    cv2.setMouseCallback(title, onMouse, mouse_params)
    cv2.imshow(title, im)

    while mouse_params['br'] is None:
        im_draw = np.copy(im)

        if mouse_params['tl'] is not None:
            cv2.rectangle(im_draw, mouse_params['tl'],
                mouse_params['current_pos'], (0, 0, 255),2)
        cv2.imshow(title, im_draw)
#        _ = cv2.waitKey(10)
        cv2.waitKey(10)
    cv2.waitKey(0)
    cv2.destroyWindow(title)

    tl = (min(mouse_params['tl'][0], mouse_params['br'][0]),
        min(mouse_params['tl'][1], mouse_params['br'][1]))
    br = (max(mouse_params['tl'][0], mouse_params['br'][0]),
        max(mouse_params['tl'][1], mouse_params['br'][1]))

    #返回矩形框坐标
    return (tl, br)  #tl=(y1,x1), br=(y2,x2)

# Process the file
# with open(args.image_list_file, 'r') as f:
#     lines = f.readlines()
# tsum=0
# for line in tqdm(lines, total=len(lines)):
#     path = line.strip()
#     print(line)
def extract_features(image,args,device,model):
    # image = cv2.resize(image,(720,576))
    #cv2.imwrite('im2.jpg',image)
    #t1=cv2.getTickCount()
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]
        image = np.repeat(image, 3, -1)

    # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
    resized_image = image

    if max(resized_image.shape) > args.max_edge:
        ratio = args.max_edge / max(resized_image.shape)
        h,w,ch = resized_image.shape
        resized_image = resize(
            resized_image,(int(h*ratio),int(w*ratio)))
        # ).astype('float')
    if sum(resized_image.shape[: 2]) > args.max_sum_edges:
        ratio = args.max_sum_edges / sum(resized_image.shape[: 2])
        h,w,ch = resized_image.shape
        resized_image = resize(
            resized_image,
            (int(h*ratio),int(w*ratio)))
        # ).astype('float')
    ()
    fact_i = image.shape[0] / resized_image.shape[0]
    fact_j = image.shape[1] / resized_image.shape[1]

    input_image = preprocess_image(
        resized_image,
        preprocessing=args.preprocessing
    )
    with torch.no_grad():
        if args.multiscale:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model
            )
        else:
            keypoints, scores, descriptors = process_multiscale(
                torch.tensor(
                    input_image[np.newaxis, :, :, :].astype(np.float32),
                    device=device
                ),
                model,
                scales=[1]
            )

    # Input image coordinates
    keypoints[:, 0] *= fact_i
    keypoints[:, 1] *= fact_j
    # i, j -> u, v
    keypoints = keypoints[:, [1, 0, 2]]
    return keypoints,scores,descriptors
    # t2=cv2.getTickCount()
    # elapsTime=(t2-t1)/cv2.getTickFrequency()
    # tsum=tsum+elapsTime
    # print('eps=%f s'%elapsTime)
    # if args.output_type == 'npz':
    #     with open(path + args.output_extension, 'wb') as output_file:
    #         np.savez(
    #             output_file,
    #             keypoints=keypoints,
    #             scores=scores,
    #             descriptors=descriptors
    #         )
    # elif args.output_type == 'mat':
    #     with open(path + args.output_extension, 'wb') as output_file:
    #         scipy.io.savemat(
    #             output_file,
    #             {
    #                 'keypoints': keypoints,
    #                 'scores': scores,
    #                 'descriptors': descriptors
    #             }
    #         )
    # else:
    #     raise ValueError('Unknown output type.')
# tsum = tsum/len(lines)
# print('mean time =%f'%tsum)
def get_xywh_xml(path):
    dom = minidom.parse(path)
    root = dom.documentElement
    xmin = root.getElementsByTagName('xmin')
    ymin = root.getElementsByTagName('ymin')
    xmax = root.getElementsByTagName('xmax')
    ymax = root.getElementsByTagName('ymax')
    if xmin == [] and ymin == [] and xmax == [] and ymax == []:
        print(path + " don't match the target!!!")
        assert ('wrong!')
        return [],[],[],[]
    xmin_l = int(xmin[0].firstChild.data)
    ymin_l = int(ymin[0].firstChild.data)
    xmax_l = int(xmax[0].firstChild.data)
    ymax_l = int(ymax[0].firstChild.data)

    return xmin_l, ymin_l, xmax_l, ymax_l

# def parse_rec(filename):
#     """ Parse a PASCAL VOC xml file """
#     tree = ET.parse(filename)
#     objects = []
#     for obj in tree.findall('object'):
#         obj_struct = {}
#         obj_struct['name'] = obj.find('name').text
#         #obj_struct['pose'] = obj.find('pose').text
#         obj_struct['truncated'] = int(obj.find('truncated').text)
#         obj_struct['difficult'] = int(obj.find('difficult').text)
#         bbox = obj.find('bndbox')
#         obj_struct['bbox'] = [int(bbox.find('xmin').text),
#                               int(bbox.find('ymin').text),
#                               int(bbox.find('xmax').text),
#                               int(bbox.find('ymax').text)]
#         objects.append(obj_struct)
#
#     return objects
#

def init_d2net():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    # Argument parsing
    parser = argparse.ArgumentParser(description='Feature extraction script')

    # parser.add_argument(
    #     '--image_list_file', type=str, required=True,
    #     help='path to a file containing a list of images to process'
    # )

    parser.add_argument(
        '--preprocessing', type=str, default='caffe',
        help='image preprocessing (caffe or torch)'
    )
    parser.add_argument(
        '--model_file', type=str, default='models/d2_tf.pth',
        help='path to the full model'
    )

    parser.add_argument(
        '--max_edge', type=int, default=1600,
        help='maximum image size at network input'
    )
    parser.add_argument(
        '--max_sum_edges', type=int, default=2800,
        help='maximum sum of image sizes at network input'
    )

    parser.add_argument(
        '--output_extension', type=str, default='.d2-net',
        help='extension for the output'
    )
    parser.add_argument(
        '--output_type', type=str, default='npz',
        help='output file type (npz or mat)'
    )

    # parser.add_argument(
    #     '--multiscale', dest='multiscale', action='store_true',
    #     help='extract multiscale features'
    # )
    # parser.set_defaults(multiscale=False)

    parser.add_argument(
        '--no-relu', dest='use_relu', action='store_false',
        help='remove ReLU after the dense feature extraction module'
    )
    parser.set_defaults(use_relu=True)

    # parser.add_argument(
    #     '--match_thresh', type=int, default=9,
    #     help='min match_thresh pair'
    # )
    #
    parser.add_argument(
        '--img_path_left', type=str,
        help='img_path_left'
    )

    parser.add_argument(
        '--img_right_folder', type=str,
        help='img_right_folder'
    )

    args = parser.parse_args()
    args.multiscale = True

    model = D2Net(
        model_file=args.model_file,
        use_relu=args.use_relu,
        use_cuda=use_cuda
    )

    return model,args,device

# if __name__ == '__main__':
def main_d2(img_left,img_right_path,model,args,device,
            keypoints_left,descriptor_left,  sorces_left,
            is_reX,is_reY):
    # CUDA

    # use_cuda = torch.cuda.is_available()
    # device = torch.device("cuda:0" if use_cuda else "cpu")
    #
    # # Argument parsing
    # parser = argparse.ArgumentParser(description='Feature extraction script')
    #
    # # parser.add_argument(
    # #     '--image_list_file', type=str, required=True,
    # #     help='path to a file containing a list of images to process'
    # # )
    #
    # parser.add_argument(
    #     '--preprocessing', type=str, default='caffe',
    #     help='image preprocessing (caffe or torch)'
    # )
    # parser.add_argument(
    #     '--model_file', type=str, default='models/d2_tf.pth',
    #     help='path to the full model'
    # )
    #
    # parser.add_argument(
    #     '--max_edge', type=int, default=1600,
    #     help='maximum image size at network input'
    # )
    # parser.add_argument(
    #     '--max_sum_edges', type=int, default=2800,
    #     help='maximum sum of image sizes at network input'
    # )
    #
    # parser.add_argument(
    #     '--output_extension', type=str, default='.d2-net',
    #     help='extension for the output'
    # )
    # parser.add_argument(
    #     '--output_type', type=str, default='npz',
    #     help='output file type (npz or mat)'
    # )
    #
    # # parser.add_argument(
    # #     '--multiscale', dest='multiscale', action='store_true',
    # #     help='extract multiscale features'
    # # )
    # # parser.set_defaults(multiscale=False)
    #
    # parser.add_argument(
    #     '--no-relu', dest='use_relu', action='store_false',
    #     help='remove ReLU after the dense feature extraction module'
    # )
    # parser.set_defaults(use_relu=True)
    #
    # # parser.add_argument(
    # #     '--match_thresh', type=int, default=9,
    # #     help='min match_thresh pair'
    # # )
    # #
    # parser.add_argument(
    #     '--img_path_left', type=str,
    #     help='img_path_left'
    # )
    #
    # parser.add_argument(
    #     '--img_right_folder', type=str,
    #     help='img_right_folder'
    # )
    #
    # args = parser.parse_args()
    # args.multiscale = True
    # # print(args)
    # # print('cnt=%d'%cnt)
    # # Creating CNN model
    # if cnt == 0:
    #     model = D2Net(
    #     model_file=args.model_file,
    #     use_relu=args.use_relu,
    #     use_cuda=use_cuda
    #     )
    # else:
    #     model =model

    match_thresh=3
    log=[]
    # pair_id=14 img_path_left img_right_folder
    # img_path_left='video/%d.jpg'%pair_id
    # img_right_folder='video/scenario_%d/imgs/'%pair_id
    # img_path_left =args.img_path_left
    # img_right_folder =args.img_right_folder
    # img_left_large = imageio.imread(img_path_left)
    # top,bottom=get_rect(img_left_large)
    # # tt = top[1]
    # img_left = img_left_large[top[1]:bottom[1],top[0]:bottom[0],:]
    # cv2.rectangle(img_left_large,top,bottom,(0,0,255),2)
    # img_left_name='video/t%d.jpg'%pair_id
    # img_left = imageio.imread(img_left_name)
    # cv2.imshow('left',img_left)
    # cv2.waitKey(0)
    # keypoints_left,  sorces_left,  descriptor_left=extract_features(img_left,args,device,model)
    # img_files =os.listdir(img_right_folder)
    # if os._exists(img_right_folder + '/result'):
    #     os.rmdir(img_right_folder + '/result')
    # os.mkdir(img_right_folder + '/result')
    # IOU_set=[]
    # for img_path_right in img_files:
    #     img_right = imageio.imread(img_right_folder+'/'+img_path_right)
    img_right=cv2.imread(img_right_path,1)
    # img_right=imageio.imread(img_right_path)
    keypoints_right, sorces_right, descriptor_right = extract_features(img_right,args,device,model)

    matches = match_descriptors(descriptor_left, descriptor_right, cross_check=True)

    # print('Number of raw matches: %d.' % matches.shape[0])
    log.append('Number of raw matches: %d.' % matches.shape[0])
    #print(keypoints_left)
    keypoints_ll = keypoints_left[matches[:, 0], : 2]
    keypoints_rr = keypoints_right[matches[:, 1], : 2]
    np.random.seed(0)
    best_model_, inliers = ransac(
        (keypoints_ll, keypoints_rr),
        ProjectiveTransform, min_samples=4,
        residual_threshold=4, max_trials=100  # res= 2 maxtrial=10000
    )
    n_inliers = np.sum(inliers)
    # print('Number of inliers: %d.' % n_inliers)
    log.append('Number of inliers: %d.' % n_inliers)
    inlier_keypoints_left = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_ll[inliers]]
    inlier_keypoints_right = [cv2.KeyPoint(point[0], point[1], 1) for point in keypoints_rr[inliers]]
    placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

    if n_inliers > match_thresh:#0.05 * matches.shape[0]:
        matched_pts1 = cv2.KeyPoint_convert(inlier_keypoints_left)
        matched_pts2 = cv2.KeyPoint_convert(inlier_keypoints_right)

        h, w, ch = img_left.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(matched_pts1,
                                        matched_pts2,
                                        cv2.RANSAC,
                                        ransacReprojThreshold=4)
        if H is not None:
            dst = cv2.perspectiveTransform(pts, H)

            cord_is_right = True
            # Compute the Ground-Truth & Our target region IOU
            # Get the object coordinate
            # dst=dst[0]
            # p0=dst[0][0][0]
            # p1=dst[0][0][1]
            # p2=dst[1][0][0]
            # p3=dst[1][0][1]
            # p4=dst[2][0][0]
            # p5=dst[2][0][1]
            # p6=dst[3][0][0]
            # p7=dst[3][0][1]
            # cv2.line(img_right, (p0,p1),(p2,p3), (255, 0, 0),2)
            # cv2.line(img_right, (p2,p3),(p4,p5), (255, 0, 0),2)
            # cv2.line(img_right, (p4,p5),(p6,p7), (255, 0, 0),2)
            # cv2.line(img_right, (p6,p7),(p0,p1), (255, 0, 0),2)
            xy=np.mat(dst)
            x_min = np.min(xy[:, 0])
            y_min = np.min(xy[:, 1])
            x_max = np.max(xy[:, 0])
            y_max = np.max(xy[:, 1])
            # if is_reX:
            #     x_min= int(x_min+50)
            #     x_max= int(x_max-50)
            # if is_reY:
            #     y_min= int(y_min+50)
            #     y_max= int(y_max-50)

            cv2.rectangle(img_right, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

            # image2xml = img_right_folder + 'outputs/DJI_%04d '%pair_id +'%03d.xml'%id
            # # objects = parse_rec(image2xml)
            # gt_xmin, gt_ymin, gt_xmax, gt_ymax = get_xywh_xml(image2xml)
            #
            # if gt_xmin == [] and gt_ymin == [] and gt_xmax == [] and gt_ymax == []:
            #     print(image2xml + " don't match the target!!!")
            #     IOU_set.append(0)
            #     continue
            # xmin = max(x_min, gt_xmin)
            # ymin = max(y_min, gt_ymin)
            # xmax = min(x_max, gt_xmax)
            # ymax = min(y_max, gt_ymax)
            #
            # s1 = (x_max - x_min) * (y_max - y_min)
            # s2 = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
            # s = s1 + s2
            # inter_area = (xmax - xmin) * (ymax - ymin)
            # iou = inter_area / (s - inter_area)
            # if iou >= 1 or iou < 0:
            #     IOU_set.append(0)
            #     print(image2xml + " IOU: 0")
            # else:
            #     IOU_set.append(iou)
            #     print(image2xml + " IOU:", iou)
            # cv2.rectangle(img_right, (gt_xmin, gt_ymin), (gt_xmax, gt_ymax), (0, 255, 0), 2)

    else:
        print('nor enough inliers%s--' + '%s' % img_right_path)
        log.append('nor enough inliers%s--' + '%s' % img_right_path)
        # IOU_set.append(0)
    # cv2.polylines(image2, [np.int32(dst)], True, (255,0,255), 1, cv2.LINE_AA)
    # image3 = cv2.drawMatches(img_left, None, img_right, None,
    #                          None, None)
    # ''' 显示匹配点连线'''
    # if n_inliers > match_thresh:
    image3 = cv2.drawMatches(img_left, inlier_keypoints_left, img_right, inlier_keypoints_right, placeholder_matches, None)
    # else:
    # image3 = cv2.drawMatches(img_left_large, None, img_right, None,
    #                               None, None)
    # cv2.putText(image3, 'raw matches %d' % matches.shape[0] + '  Number of inliers: %d.' % n_inliers,
    #             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    # pathname = img_right_folder+'/result/'+img_path_right
    # cv2.imshow('result', image3)
    # cv2.waitKey(0)
    # cv2.imwrite(pathname, img_right)
    return img_right,log
    # cv2.imshow('result',image3)
        #
    # tmp_IOU = 0
    # for i in range(len(IOU_set)-1):
    #     tmp_IOU = tmp_IOU+IOU_set[i]
    # if len(IOU_set)!=1:
    #     average_iou = tmp_IOU / (len(IOU_set)-1)
    #     ss2=0
    #     for i in range(len(IOU_set)-1):
    #         ss2 = ss2+(IOU_set[i]-average_iou)*(IOU_set[i]-average_iou)
    #     ss2 = math.sqrt(ss2 / (len(IOU_set) - 1))
    # print(IOU_set)
    #
    # pl.plot(np.arange(1,len(IOU_set)+1,1), IOU_set)
    # pl.xlabel("Image number")
    # pl.ylabel("IOU value")
    # pl.title("ATR images IOU diagram")
    # pl.savefig(img_right_folder+ 'results/'+ 'iou.jpg')
    # pl.close()
    # # pl.show()
    # # IOU_set.append('avg:')
    # IOU_set.append(average_iou)
    # # IOU_set.append('ss2:')
    # # IOU_set.append(ss2)
    # np.savetxt(img_right_folder+'results/'+'iou.txt',IOU_set)
    # print(average_iou)