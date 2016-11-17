import numpy as np
import cv2
from matplotlib import pyplot as plt


def proj(img, mask, factor=0.25):
    size = img.shape
    size = [size[1], size[0]]
    anchor_offset = np.random.rand(4,2) * factor
    anchor_offset[:,0] *= size[0]
    anchor_offset[:,1] *= size[1]
    corner = np.array([[0,0], [0,size[1]], [size[0], size[1]], [size[0],0]])
    new_corner = np.array([[size[0]*0+anchor_offset[0,0], size[1]*0+anchor_offset[0,1]],
                           [size[0]*0+anchor_offset[1,0], size[1]*1-anchor_offset[1,1]],
                           [size[0]*1-anchor_offset[2,0], size[1]*1-anchor_offset[2,1]],
                           [size[0]*1-anchor_offset[3,0], size[1]*0+anchor_offset[3,1]]])
    M = cv2.getPerspectiveTransform(corner.astype(np.float32), new_corner.astype(np.float32))
    new_img = cv2.warpPerspective(img, M, (size[0]*1, size[1]*1), flags=cv2.INTER_CUBIC)
    new_mask = cv2.warpPerspective(mask, M, (size[0]*1, size[1]*1), flags=cv2.INTER_NEAREST)
    return new_img, new_mask


def crop(img, mask, size, center=True, border=10):
    np_img = np.array(img)
    np_mask = np.array(mask)
    left = 0
    right = np_img.shape[0]-1
    top = 0
    bottom = np_img.shape[1]-1 
    while not np.sum(np_img[left,:]):
        left += 1
    while not np.sum(np_img[right,:]):
        right -= 1
    while not np.sum(np_img[:,top]):
        top += 1
    while not np.sum(np_img[:,bottom]):
        bottom -= 1
    np_img = np_img[left:right+1, top:bottom+1]
    np_mask= np_mask[left:right+1, top:bottom+1]
    factor = min(1.*(size[0]-border)/np_img.shape[0], 1.*(size[1]-border)/np_img.shape[1])
    if factor < 1:
        np_img = cv2.resize(np_img, (int(factor*np_img.shape[1]), int(factor*np_img.shape[0])), interpolation=cv2.INTER_CUBIC)
        np_mask = cv2.resize(np_mask, (int(factor*np_mask.shape[1]), int(factor*np_mask.shape[0])), interpolation=cv2.INTER_NEAREST)
    width, height = np_img.shape[:2]
    if center:
        lcrop = (size[0]-width) / 2
        tcrop = (size[1]-height) /2
    else:
        lcrop = np.random.randint(size[0]-width)
        tcrop = np.random.randint(size[1]-height)
    new_img = np.zeros([size[0],size[1],3], np.uint8)
    new_img[lcrop:lcrop+width, tcrop:tcrop+height, :] = np_img
    new_mask = np.zeros([size[0],size[1],3], np.uint8)
    new_mask[lcrop:lcrop+width, tcrop:tcrop+height, :] = np_mask
    return new_img, new_mask


def spherical(img, mask, pos=[], scale=[]):
    size = img.shape
    if not scale:
        scale = [np.pi/2, np.pi/2]
    if not pos:
        pos = [0, 0]
    pos[0] = scale[0] - pos[0]
    pos[1] = scale[1] - pos[1]
    scale[0] = size[0]*.5 / scale[0] # old coor' = old coor / scale - pos
    scale[1] = size[1]*.5 / scale[1]
    # old region
    ymin, ymax = -pos[1], size[1]/scale[1]-pos[1]
    xmin, xmax = -pos[0], size[0]/scale[0]-pos[0]
    # new_y = sin(y)
    # new_x = sin(x)*cos(y)
    siny_max = np.sin(ymax)
    siny_min = np.sin(ymin)
    sinx_max = np.sin(xmax)
    sinx_min = np.sin(xmin)
    if ymin <= 0 and ymax >= 0:
        cosy_max = 1
    else:
        cosy_max = max(np.cos(ymax), np.cos(ymin))
    cosy_min = min(np.cos(ymax), np.cos(ymin))
    ymin = siny_min
    ymax = siny_max
    xmin = min(sinx_min*cosy_min, sinx_min*cosy_max)
    xmax = max(sinx_max*cosy_min, sinx_max*cosy_max)

    scale2 = [0, 0] # new coor' = new coor / scale2 - pos2
    pos2 = [0, 0]
    scale2[0] = 1. * size[0] / (xmax-xmin)
    scale2[1] = 1. * size[1] / (ymax-ymin)
    pos2[0] = -xmin
    pos2[1] = -ymin

    map = np.zeros([size[0], size[1], 2], np.float32)
    # y' = u' / scale2 - pos2
    # x' = v' / scale2 - pos2
    # y' = sin(y)
    # x' = sin(x)cos(y)
    # x = u / scale - pos
    # y = v / scale - pos
    # 
    # y = arcsin(y')
    # x = arcsin(x'/(sqrt(1-y'^2)))
    #
    # u = f(u', v'), v = g(u', v')
    for i in range(size[0]):
        for j in range(size[1]):
            x2 = i / scale2[0] - pos2[0]
            y2 = j / scale2[1] - pos2[1]
            x = np.arcsin(x2/np.sqrt(1-y2**2)) 
            y = np.arcsin(y2)
            u = (x+pos[0]) * scale[0]
            v = (y+pos[1]) * scale[1]
            map[i,j,0] = u
            map[i,j,1] = v
    return cv2.remap(img, map[:,:,1], map[:,:,0], cv2.INTER_CUBIC), cv2.remap(mask, map[:,:,1], map[:,:,0], cv2.INTER_NEAREST)
    






