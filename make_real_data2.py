import time
import random
import os
import mxnet as mx
import numpy as np
import random
import string
import cPickle as pickle
np.set_printoptions(precision=2)

from PIL import Image, ImageFont, ImageDraw, ImageFilter
from transform import *
from skimage import io, transform, morphology


VGGPATH = './vgg19.params'


def get_font(font_path='/home/zw/dataset/fonts/fonts/OpenSans-Regular.ttf', text=['qwerty'], cls={}): 
    font = ImageFont.truetype(font_path, 160)
    img = Image.new("RGB", (2000, 1000),(0,0,0))
    draw = ImageDraw.Draw(img)
    y_start = [50]
    x_start = []

    for line in text:
        x_start.append(500 + np.random.randint(-100,100))
        max_height = 0
        tmp = x_start[-1]
        for char in line:
            draw.text((tmp, y_start[-1]), char, (255,255,255), font=font)
            tmp += font.getsize(char)[0]
            max_height = max(max_height, font.getsize(char)[1])
        y_start.append(y_start[-1]+max_height+np.random.randint(0,100))
    y_start.pop()
    img = np.array(img)
    img[:,:,1] = morphology.binary_dilation(img[:,:,1], morphology.disk(np.random.randint(3,6)**2))*255
#    img = Image.fromarray(img)
#    img = img.filter(ImageFilter.GaussianBlur(3))
#    mask = Image.new("RGB", (2000, 400),(0,0,0))
#    draw = ImageDraw.Draw(mask)
    mask = np.zeros([1000,2000,3], np.uint8)
    for x,y,line in zip(x_start, y_start, text):
        xt = x
        for char in line:
            mask[y+font.getoffset(char)[1]:y+font.getoffset(char)[1]+font.getsize(char)[1],xt+font.getoffset(char)[0]:xt+font.getoffset(char)[0]+font.getsize(char)[0],:] = cls[char]
            xt += font.getsize(char)[0]+font.getoffset(char)[0]
    img, mask = crop(img, mask, [512, 1024])
    t = time.time()
    while True:
        pos0 = np.random.rand()**2*1.5 - .75
        pos1 = np.random.rand()**2*1.5 - .75
        scale0 = np.random.rand()**2*.5 + .25
        scale1 = np.random.rand()**2*.5 + .25
        if abs(pos0)+abs(scale0) > 1.25 or abs(pos1)+abs(scale1) > 1.25:
            continue
        img, mask = spherical(img, mask, pos=[pos0,pos1], scale=[scale0, scale1])
        break
    img, mask = crop(img, mask, [384, 768], border=10)
    img, mask = proj(img, mask, 0.25)
    img, mask = crop(img, mask, [96, 192], border=0, center=True)
    img = Image.fromarray(img)
    img = img.filter(ImageFilter.GaussianBlur(1))
    img = np.array(img)
#    img[:,:,0] *= np.random.randint(255)
#    img[:,:,1] *= np.random.randint(255)
#    img[:,:,2] *= np.random.randint(255)
    return img, mask


def test():
    def preprocess_img(im, size):
#        im = crop_img(im, size)
        im = im.astype(np.float32)
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
        im[0,:] -= 123.68
        im[1,:] -= 116.779
        im[2,:] -= 103.939
        im = np.expand_dims(im, 0)
        return im
    def crop_img(im, size):
        if im.shape[0] > im.shape[1]:
            c = (im.shape[0]-im.shape[1]) / 2
            im = im[c:c+im.shape[1],:,:]
        else:
            c = (im.shape[1]-im.shape[0]) / 2
            im = im[:,c:c+im.shape[0],:]
        im = transform.resize(im, (size[1],size[0]))
        return im
    def postprocess_img(im):
        im = im[0]
        im[0,:] += 123.68
        im[1,:] += 116.779
        im[2,:] += 103.939
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 0, 1)
        im[im<0] = 0
        im[im>255] = 255
        return im.astype(np.uint8)


    font_dir = '/home/zw/dataset/fonts/fonts/'
    img_dir = '/home/zw/dataset/mscoco/'
    font_d = os.listdir(font_dir)
    img_d = os.listdir(img_dir)
    try:
        with open('bold_font.pkl') as f:
            bold_font = pickle.load(f)
        with open('all_font.pkl') as f:
            all_font = pickle.load(f)
    except:
        bold_font = []
        all_font = []
        for font in font_d:
            font_path = os.path.join(font_dir, font)
            font = ImageFont.truetype(font_path, 60)
            img = Image.new("RGB", (200, 100),(0,0,0))
            draw = ImageDraw.Draw(img)
            draw.text((0,0), 'a', (255,255,255), font=font)
            img = np.array(img)
            if img.sum() > 300*255*3:
                bold_font.append(font_path)
            all_font.append(font_path)
        with open('bold_font.pkl', 'w') as f:
            pickle.dump(bold_font, f)
        with open('all_font.pkl', 'w') as f:
            pickle.dump(all_font, f)

    cls = {}
    chars = string.letters+string.digits
    for (i,c) in enumerate(chars):
        cls[c] = i+1
    count = 0
    while count < 10000:
        t = time.time()

        scbg = np.random.rand() > 0.7
        scfg = scbg or np.random.rand() > 0.5

        if not scbg:
            while True:
                bg_img = io.imread(img_dir+random.choice(img_d))
                if bg_img == None:
                    continue
                bg_size = bg_img.shape
                if bg_size[0] < 384 or bg_size[1] < 384 or len(bg_size) != 3:
                    continue
                bg_img = crop_img(bg_img, [384,384])
                break
            if bg_img.shape != (384,384,3):
                continue
            bg_img *= 255
        else:
            bg_img = np.ones([384,384,3])
            for i in range(3):
                bg_img[:,:,i] *= np.random.randint(0, 255)
            bg_img = bg_img.astype(np.float)
        anno = np.zeros_like(bg_img)

        for ii in range(4):
            for jj in range(2):
                padding = np.random.rand() > 0.75 and scfg
                if padding or not scfg:
                    need_bold = 1
                else:
                    need_bold = 0
                lines = np.random.randint(1,8)
                text = []
                for i in range(lines):
                    text.append(''.join([random.choice(chars) for i in range(random.randint(5,20))]))
                if need_bold:
                    font_path = random.choice(bold_font)
                else:
                    font_path = random.choice(all_font)
                image, mask = get_font(font_path, text, cls)
                padding_image = np.expand_dims(image[:,:,1], 2)
                padding_image = np.tile(padding_image, [1,1,3])
                if not padding:
                    padding_image[:] = 0
                image[:,:,1] = image[:,:,0]
                image[:,:,2] = image[:,:,0]
                bg_avg = bg_img[96*ii:96*ii+96,192*jj:192*jj+192,:].mean(axis=0).mean(axis=0)
                if not scfg:
                    while True:
                        fg_img = io.imread(img_dir+random.choice(img_d))
                        if fg_img == None:
                            continue
                        fg_size = fg_img.shape
                        if fg_size[0] < 97 or fg_size[1] < 193 or len(fg_size) != 3:
                            continue
                        xstart = np.random.randint(0, fg_size[0]-96)
                        ystart = np.random.randint(0, fg_size[1]-192)
                        fg_img = fg_img[xstart:xstart+96,ystart:ystart+192,:].astype(np.float)
                        if fg_img.shape != (96,192,3):
                            continue
                        fg_avg = fg_img.mean(axis=0).mean(axis=0)
                        if np.sum(np.abs(fg_avg-bg_avg)) < 200:
                            continue
                        break
                else:
                    while True:
                        fg_color = np.random.randint(0,255,[3])
                        if np.abs(fg_color-bg_avg).sum() < 200:
                            continue
                        break
                    fg_img = np.ones([96,192,3])
                    for i in range(3):
                        fg_img[:,:,i] = fg_color[i]
                    fg_avg = fg_color
                while True:
                    pd_color = np.random.randint(0,255,[1,1,3])
                    if np.abs(fg_avg-pd_color).sum() < 200:
                        continue
                    break

                enhance = (fg_avg>bg_avg) * 255
                fg_img[:,:,0] = fg_img[:,:,0]*0.75+enhance[0]*0.25
                fg_img[:,:,1] = fg_img[:,:,1]*0.75+enhance[1]*0.25
                fg_img[:,:,2] = fg_img[:,:,2]*0.75+enhance[2]*0.25
                image = image.astype(np.float)/255
                padding_image = padding_image.astype(np.float)/255
                image[image<0.5] = 0
                image[image>0.5] = 1
                padding_image[padding_image<0.5] = 0
                padding_image[padding_image>0.5] = 1
                bg_img[96*ii:96*ii+96,192*jj:192*jj+192,:] = bg_img[96*ii:96*ii+96,192*jj:192*jj+192,:]*(1-padding_image)+pd_color*padding_image
                bg_img[96*ii:96*ii+96,192*jj:192*jj+192,:] = bg_img[96*ii:96*ii+96,192*jj:192*jj+192,:]*(1-image)+fg_img*image
                anno[96*ii:96*ii+96,192*jj:192*jj+192,:] = mask
        
        img = Image.fromarray(bg_img.astype(np.uint8))
        img.save('/home/zw/dataset/scene_text2/%d.jpg'%(count), quality=100)
        anno = anno[:,:,0].astype(np.uint8)
        anno = Image.fromarray(anno.astype(np.uint8))
        anno.save('/home/zw/dataset/scene_text2/%d.png'%(count), quality=100)
        print count 
        count += 1

if __name__ == '__main__':
    test()
