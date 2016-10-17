import PIL
import cv2
import os
import numpy as np

from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw

def crop(img):
    np_img = np.array(img)
    left = 0
    right = 127 
    top = 0
    bottom = 127
    while not np.sum(np_img[left,:]):
        left += 1
    while not np.sum(np_img[right,:]):
        right -= 1
    while not np.sum(np_img[:,top]):
        top += 1
    while not np.sum(np_img[:,bottom]):
        bottom -= 1
    np_img = np_img[left:right+1, top:bottom+1]
    factor = 60. / max(np_img.shape)
    np_img = cv2.resize(np_img, (int(factor*np_img.shape[1]), int(factor*np_img.shape[0])), cv2.INTER_NEAREST)
    width, height = np_img.shape[:2]
    lcrop = (64-width) / 2
    tcrop = (64-height) /2
    new_img = np.zeros([64,64,3], np.uint8)
    new_img[lcrop:lcrop+width, tcrop:tcrop+height, :] = np_img
    return new_img


def skew(img):
    anchor_offset = np.random.randint(0, 32, [4,2])
    corner = np.array([[0,0], [0,64], [64,64], [64,0]])
    new_corner = np.array([[0+anchor_offset[0,0], 0+anchor_offset[0,1]],
                           [0+anchor_offset[1,0], 127-anchor_offset[1,1]],
                           [127-anchor_offset[2,0], 127-anchor_offset[2,1]],
                           [127-anchor_offset[3,0], 0+anchor_offset[3,1]]])
    M, mask = cv2.findHomography(corner, new_corner)
    new_img = cv2.warpPerspective(img, M, (128,128))
    return new_img


def save_img(font, font_name, char, background=''):
    img = Image.new("RGB", (128, 128),(0,0,0))
    draw = ImageDraw.Draw(img)
    draw.text((50, 40), char, (255,255,255), font=font)
    ImageDraw.Draw(img)
    img = crop(img)
    img = skew(img)
    img = crop(img)
    img = Image.fromarray(255-img)
    img.save('%s-%s.jpg'%(background, chars))


ord0 = ord('0')
orda = ord('a')
ordA = ord('A')
# os.mkdir('font_pics')
# for i in range(10):
#     os.mkdir('font_pics/'+chr(ord0+i))
# for i in range(26):
#     os.mkdir('font_pics/'+chr(orda+i))
# for i in range(26):
#     os.mkdir('font_pics/'+chr(ordA+i))


for d in os.listdir('/home/zw/dataset/fonts/fonts'):
    print d
    font = ImageFont.truetype("/home/zw/dataset/fonts/fonts/%s"%d, 40)
    font_name = d.split('.')[0]
    for j in range(10):
        for i in range(10):
            char = chr(ord0+i)
            save_img(font, font_name+str(j), char)
        for i in range(26):
            char = chr(orda+i)
            save_img(font, font_name+str(j), char)
        for i in range(26):
            char = chr(ordA+i)
            save_img(font, font_name+str(j), char)
