import numpy as np
global _strict
_strict = True


def RGBT_start():

    global _strict
    _strict = False

def RGBT_end():

    global _strict
    _strict = True


def load_text(path, dtype=np.float32):
    return load_text_numpy(path, delimiter=[',', ' ', '\t'], dtype=dtype)


def load_text_numpy(path, delimiter=' ', dtype=np.float32):
    if isinstance(delimiter, (tuple, list)):
        for d in delimiter:
            try:
                ground_truth_rect = np.loadtxt(path, delimiter=d, dtype=dtype)
                return ground_truth_rect
            except:
                pass

        raise Exception('Could not read file {}'.format(path))
    else:
        ground_truth_rect = np.loadtxt(path, delimiter=delimiter, dtype=dtype)
        return ground_truth_rect


def ltwh_2_ltrb(rect):
    return [rect[0], rect[1], rect[2]+rect[0]-1, rect[3]+rect[1]-1]


def ltrb_2_ltwh(rect):
    if _strict:
        return [rect[0], rect[1], rect[2]-rect[0]+1, rect[3]-rect[1]+1]
    else:
        return [rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]]


def xywh_2_ltrb(rect):
    """
    (cx, cy, w, h) -> (left, top, right, bottom)
    """
    delta_w = (rect[2]-1)/2.
    delta_h = (rect[3]-1)/2.
    return [rect[0]-delta_w, rect[1]-delta_h, rect[0]+delta_w, rect[1]+delta_h]


def xywh_2_ltwh(rect):
    """
    (cx, cy, w, h) -> (left, top, width, height)
    """
    delta_w = (rect[2]-1)/2.
    delta_h = (rect[3]-1)/2.
    return [rect[0]-delta_w, rect[1]-delta_h, rect[2], rect[3]]


def corner_2_ltrb(rect):
    return [rect[0], rect[1], rect[-2-2], rect[-1-2]]


def corner_2_ltwh(rect):
    if _strict:
        return [rect[0], rect[1], rect[2]-rect[0]+1, rect[5]-rect[1]+1]
    else:
        return [rect[0], rect[1], rect[2]-rect[0], rect[5]-rect[1]]


def bbox_type_trans(bbox_type_src, bbox_type_new):
    if bbox_type_src!=bbox_type_new:
        return eval(bbox_type_src+'_2_'+bbox_type_new)
    else:
        return lambda x:x


def serial_process(fun, *serial):
    return list(map(fun, *serial))


def normalize(cx, cy, gt_w, gt_h, eps=1e-8):
    return cx/(gt_w+eps), cy/(gt_h+eps)


def CLE(rect1, rect2):
    """ caculate center location error
    NOTE: Default rect2 is groundtruth
    Args:
        rect1: (l, t, w, h)
        rect2: (l, t, w, h)
    Returns:
        center location error
    """
    if _strict:
        cp1 = [(rect1[2]-1)/2.+rect1[0], (rect1[3]-1)/2.+rect1[1]]
        cp2 = [(rect2[2]-1)/2.+rect2[0], (rect2[3]-1)/2.+rect2[1]]
    else:
        cp1 = [rect1[2]/2.+rect1[0], rect1[3]/2.+rect1[1]]
        cp2 = [rect2[2]/2.+rect2[0], rect2[3]/2.+rect2[1]]
    d = ((cp1[0]-cp2[0])**2 + (cp1[1]-cp2[1])**2)**0.5
    return d


def normalize_CLE(rect1, rect2):
    """ caculate center location error
    NOTE: Default rect2 is groundtruth
    Args:
        rect1: (l, t, w, h)
        rect2: (l, t, w, h)
    Returns:
        center location error
    """
    if _strict:
        cp1 = [(rect1[2]-1)/2.+rect1[0], (rect1[3]-1)/2.+rect1[1]]
        cp2 = [(rect2[2]-1)/2.+rect2[0], (rect2[3]-1)/2.+rect2[1]]
    else:
        cp1 = [rect1[2]/2.+rect1[0], rect1[3]/2.+rect1[1]]
        cp2 = [rect2[2]/2.+rect2[0], rect2[3]/2.+rect2[1]]
    cp1 = normalize(cp1[0], cp1[1], rect2[2], rect2[3])
    cp2 = normalize(cp2[0], cp2[1], rect2[2], rect2[3])
    d = ((cp1[0]-cp2[0])**2 + (cp1[1]-cp2[1])**2)**0.5
    return d


def IoU(rect1, rect2):
    """ caculate interection over union
    Args:
        rect1: (l, t, w, h)
        rect2: (l, t, w, h)
    Returns:
        iou
    """

    x1, y1, x2, y2 = ltwh_2_ltrb(rect1)
    tx1, ty1, tx2, ty2 = ltwh_2_ltrb(rect2)

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1 +1)
    hh = np.maximum(0, yy2 - yy1 +1)

    area = rect1[-1]*rect1[-2]
    target_a = rect2[-1]*rect2[-2]
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou
