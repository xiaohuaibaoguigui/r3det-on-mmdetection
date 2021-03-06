from multiprocessing import Pool

import numpy as np
import torch

from mmdet.ops import polygon_iou
from .mean_ap import average_precision

import mmcv
from mmcv.utils import print_log
from terminaltables import AsciiTable

import math
import cv2
from scipy.spatial.distance import pdist

from boxx import loga, mapmp, tree, p





def rdets2points(rbboxes):
    """Convert detection results to a list of numpy arrays.

    Args:
        rbboxes (np.ndarray): shape (n, 6), xywhap encoded

    Returns:
        rbboxes (np.ndarray): shape (n, 9), x1y1x2y2x3y3x4y4p
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    prob = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, prob], axis=-1)


def polygon_overlaps(polygons1, polygons2):
    p1 = torch.tensor(polygons1[:, :8], dtype=torch.float64)  # in case the last element of a row is the probability
    p2 = torch.tensor(polygons2[:, :8], dtype=torch.float64)  # in case the last element of a row is the probability
    return polygon_iou(p1, p2).numpy()

def angleme(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    a1 = math.atan2(dy1, dx1)
    angle1 = int(a1 * 180/math.pi)

    a2 = math.atan2(dy2, dx2)
    angle2 = int(a2 * 180/math.pi)
    
    a3 = a2 - a1
    
    if a3 > math.pi:
        a3 = a3 - 2 * math.pi
    if a3 < -math.pi:
        a3 = a3 + 2 * math.pi
        
    #angle3 = int(a3 * 180/math.pi)
    angle3 = a3 * 180/math.pi
    
    if angle3 > 0:
        angle3 = 360 - angle3
    if angle3 < 0:
        angle3 = 0 - angle3
        
    return angle3


def backward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            tmp_v = [box[-2], box[-1], box[0], box[1]]
            tmp_c = [box[0],  box[1],  box[2], box[3]]
            
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            
            real_h = pdist([(tmp_v[0],tmp_v[1]), (tmp_v[2],tmp_v[3])])[0]
            real_w = pdist([(tmp_c[0],tmp_c[1]), (tmp_c[2],tmp_c[3])])[0]
            
            theta = angleme(tmp_v, [0,1,0,0])
#             if theta == 0:
#                 theta = 360

            boxes.append([x, y, real_w, real_h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            tmp_v = [box[-2], box[-1], box[0], box[1]]
            tmp_c = [box[0],  box[1],  box[2], box[3]]
            
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            
            real_h = pdist([(tmp_v[0],tmp_v[1]), (tmp_v[2],tmp_v[3])])[0]
            real_w = pdist([(tmp_c[0],tmp_c[1]), (tmp_c[2],tmp_c[3])])[0]
            
            theta = angleme(tmp_v, [0,1,0,0])
#             if theta == 0:
#                 theta = 360

            boxes.append([x, y, real_w, real_h, theta])

    return np.array(boxes, dtype=np.float32)

def direction2angle(label_text, ag):
    ag = ag / math.pi * 180
    
    if "RUP" in label_text:
        ag += 90
    elif "RDOWN" in label_text:
        ag += 180
    elif "LDOWN" in label_text:
        ag += 270
    else:
        ag += 360

    if ag < 0:
        ag += 360
        print("negative:{}".format(ag) )
            
    return ag

def rtpfp_default(det_bboxes,
                  det_angles,
                  gt_bboxes,
                  gt_bboxes_ignore=None,
                  iou_thr=0.5,
                  area_ranges=None,
                  class_name=None):
    """Check if detected bboxes are true positive or false positive.

    Args:
        det_bboxes (ndarray): Detected bboxes of this image, of shape (m, 9).
        det_angles (ndarray): Detected bbox center and angle of this image, of shape (m, 6).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 8).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 8). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.

    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))  # (n+k, 8)

    num_dets = det_bboxes.shape[0]  # m
    num_gts = gt_bboxes.shape[0]  # n
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)  # 1
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)  # (1, m)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)  # (1, m)
    detail_dic = {
        'angle_shift':[],
        'fn_counter':0,
    }

    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:  # n==0
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            raise NotImplementedError
            det_areas = det_bboxes[:, 2] * det_bboxes[:, 3]
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp, detail_dic
    ious = polygon_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            raise NotImplementedError
            gt_areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            if ious_max[i] >= iou_thr:
                matched_gt = ious_argmax[i]
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    if not gt_covered[matched_gt]:
                        gt_covered[matched_gt] = True
                        tp[k, i] = 1
                        # for every FP and there coresponsed GT
                        # i is DT index and matched_gt is GT index
                        dt_angle = direction2angle(class_name, det_angles[i][-2])
                        gt_angle = backward_convert([gt_bboxes[matched_gt]], with_label=False)[0][4]
                        detail_dic['angle_shift'].append(abs(dt_angle - gt_angle))
                        #print("GT:{:.02f}, DT:{:.02f}".format(gt_angle, dt_angle))
                    else:
                        fp[k, i] = 1
                # otherwise ignore this detected bbox, tp = 0, fp = 0
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :5]
                area = bbox[2] * bbox[3]
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
        #
        detail_dic['fn_counter'] = np.where(gt_covered == False)[0].shape[0]
    return tp, fp, detail_dic


def rget_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [rdets2points(img_res[class_id]) for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['polygons'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['polygons_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(torch.empty((0, 8), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore

def rget_cls_results_angle(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.

    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
        class_id (int): ID of a specific class.

    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [rdets2points(img_res[class_id]) for img_res in det_results]
    cls_dets_angle = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == class_id
        cls_gts.append(ann['polygons'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == class_id
            cls_gts_ignore.append(ann['polygons_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(torch.empty((0, 8), dtype=torch.float64))

    return cls_dets, cls_gts, cls_gts_ignore, cls_dets_angle


def reval_map(det_results,
              annotations,
              scale_ranges=None,
              iou_thr=0.5,
              dataset=None,
              logger=None,
              nproc=4,
              img_names=None,
              score_thr=0.7):
    """Evaluate mAP of a dataset.

    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:

            - `polygons`: numpy array of shape (n, 8)
            - `labels`: numpy array of shape (n, )
            - `polygons_ignore` (optional): numpy array of shape (k, 8)
            - `labels_ignore` (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.

    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0] ** 2, rg[1] ** 2) for rg in scale_ranges]
                   if scale_ranges is not None else None)
    
    sub_det_results = []
    for img_det in det_results:
        tmp_list = []
        for cate_det in img_det:
            if not cate_det.shape[0] == 0:
                scores = cate_det[:, -1]
                inds = scores > score_thr
                cate_det = cate_det[inds, :]
            tmp_list.append(cate_det)
        sub_det_results.append(tmp_list)
        
    sub_det_results = det_results
    
    eval_results = []
    angle_shift_results = []
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore, cls_dets_angle = rget_cls_results_angle(
            sub_det_results, annotations, i)
        
        mytp = []
        myfp = []
        angle_shift = []
        # 将 FN 信息添加到输入表
        num_fns = np.zeros(num_scales, dtype=int)
        num_angle_shift = np.zeros(num_scales, dtype=float)
        for cdt, cgt, cgti, cdta, iout, arear, cname, img_name in \
                                zip(cls_dets, cls_gts, cls_gts_ignore, cls_dets_angle,
                                              [iou_thr for _ in range(num_imgs)],
                                              [area_ranges for _ in range(num_imgs)],
                                              [dataset[i] for _ in range(num_imgs)],
                                              img_names):
            tp, fp, detail_dic = rtpfp_default(cdt, cdta, cgt, cgti, iout, arear, cname)
            mytp.append(tp)
            myfp.append(fp)
            
            # 打印较大的角度偏移
            ashift = detail_dic['angle_shift']
            angle_shift.extend(ashift)
            shift_ = np.array(ashift).mean() if not len(ashift) == 0 else 0.
#             if shift_ > 10.0:
#                 print("cate:{}, shift:{:.02f}, name:{}".format(dataset[i], shift_, img_name))
            # 打印漏检样本
            fn_num = detail_dic['fn_counter']
            num_fns[0] += fn_num
#             if fn_num > 0:
#                 print("cate:{}, FN:{}, name:{}".format(dataset[i], fn_num, img_name))
            
        cur_angle_shift = round(np.array(angle_shift).mean(), 2)
        angle_shift_results.append(cur_angle_shift)
        num_angle_shift[0] = cur_angle_shift
        
        mytp = tuple(mytp)
        myfp = tuple(myfp)
        tp, fp = mytp, myfp
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = bbox[:, 2] * bbox[:, 3]
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        tp = np.hstack(tp)[:, sort_inds]
        fp = np.hstack(fp)[:, sort_inds]
        # calculate recall and precision with tp and fp
        tp = np.cumsum(tp, axis=1)
        fp = np.cumsum(fp, axis=1)
        eps = np.finfo(np.float32).eps
        recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
        precisions = tp / np.maximum((tp + fp), eps)
        # calculate AP
        if scale_ranges is None:
            recalls = recalls[0, :]
            precisions = precisions[0, :]
            num_gts = num_gts.item()
            num_fns = num_fns.item()
            num_ass = num_angle_shift.item()
        mode = '11points'  # at least for DOTA dataset we use 11 points
        ap = average_precision(recalls, precisions, mode)
        eval_results.append({
            'num_gts': num_gts,
            'num_dets': num_dets,
            'num_fns': num_fns,
            'recall': recalls,
            'precision': precisions,
            'ap': ap,
            'num_ass': num_ass,
        })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results])
        all_num_gts = np.vstack(
            [cls_result['num_gts'] for cls_result in eval_results])
        mean_ap = []
        for i in range(num_scales):
            if np.any(all_num_gts[:, i] > 0):
                mean_ap.append(all_ap[all_num_gts[:, i] > 0, i].mean())
            else:
                mean_ap.append(0.0)
    else:
        aps = []
        for cls_result in eval_results:
            if cls_result['num_gts'] > 0:
                aps.append(cls_result['ap'])
        mean_ap = np.array(aps).mean().item() if aps else 0.0
    
#     print("**Angle Shift**")
#     for cname, asf in zip(dataset, angle_shift_results):
#         print("{}:{:.02f}".format(cname, asf))
#     print("Average Angle Shift:", np.array(angle_shift_results).mean())
    print_map_summary(
        mean_ap, eval_results, dataset, area_ranges, logger=logger)

    return mean_ap, eval_results

def print_map_summary(mean_ap,
                      results,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.

    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.

    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[0]['ap'], np.ndarray):
        num_scales = len(results[0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results)

    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    num_fns = np.zeros((num_scales, num_classes), dtype=int)
    num_ass = np.zeros((num_scales, num_classes), dtype=float)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
        aps[:, i] = cls_result['ap']
        num_gts[:, i] = cls_result['num_gts']
        num_fns[:, i] = cls_result['num_fns']
        num_ass[:, i] = cls_result['num_ass']

    if dataset is None:
        label_names = [str(i) for i in range(num_classes)]
    elif mmcv.is_str(dataset):
        label_names = get_classes(dataset)
    else:
        label_names = dataset

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'FNs', 'recall', 'ap', 'AS']
    for i in range(num_scales):
        if scale_ranges is not None:
            print_log(f'Scale range {scale_ranges[i]}', logger=logger)
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                label_names[j], num_gts[i, j], results[j]['num_dets'], num_fns[i,j],
                f'{recalls[i, j]:.3f}', f'{aps[i, j]:.3f}', num_ass[i,j]
            ]
            table_data.append(row_data)
        mean_as = round(np.array(num_ass[i]).mean(),2)
        table_data.append(['mAP', '', '', f'{sum(num_fns[i])}', '', f'{mean_ap[i]:.3f}', str(mean_as)])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
