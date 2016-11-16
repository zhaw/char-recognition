import time
import random
import os
import mxnet as mx
import numpy as np
np.set_printoptions(precision=2)
import cv2
from texture import symbol

from PIL import Image, ImageFont, ImageDraw, ImageFilter
from transform import *


VGGPATH = './vgg19.params'



class Maker():

    def __init__(self, st, size):
        self.size = size
        s0, s1 = size
        s0 += 128
        s1 += 128
        generator = symbol.generator_symbol(5, 'texture')
        args = mx.nd.load('texture/argst%s.jpg_texture.nd'%st)
        for i in range(5):
            args['z_%d'%i] = mx.nd.zeros([1,3,s0/16*2**i,s1/16*2**i], mx.gpu())
        self.gene_executor = generator.bind(ctx=mx.gpu(), args=args, aux_states=mx.nd.load('texture/auxst%s.jpg_texture.nd'%st))
    
    @staticmethod
    def postprocess_img(im):
        im = im[0]
        im[0,:] += 123.68
        im[1,:] += 116.779
        im[2,:] += 103.939
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 0, 1)
        im[:,:,0] *= np.random.uniform(0.1, 2)
        im[:,:,1] *= np.random.uniform(0.1, 2)
        im[:,:,2] *= np.random.uniform(0.1, 2)
        im[im<0] = 0
        im[im>255] = 255
        return im.astype(np.uint8)

    def generate(self): 
        s0, s1 = self.size
        s0 += 128
        s1 += 128
        for i in range(5):
            self.gene_executor.arg_dict['z_%d'%i][:] = np.random.uniform(-128,128,[1,3,s0/16*2**i,s1/16*2**i])
        self.gene_executor.forward()
        out = self.gene_executor.outputs[0].asnumpy()
        im = self.postprocess_img(out)
        return im[64:-64,64:-64,:]


def get_font(font_path='/home/zw/dataset/fonts/fonts/OpenSans-Regular.ttf', text='qwerty', cls={}): 
    font = ImageFont.truetype(font_path, 160)
    img = Image.new("RGB", (2000, 400),(0,0,0))
    draw = ImageDraw.Draw(img)
    start = [250, 100]
    for char in text:
        draw.text((start[0], start[1]), char, (255,255,255), font=font)
        start[0] += font.getsize(char)[0]
    img = img.filter(ImageFilter.GaussianBlur(3))
#    mask = Image.new("RGB", (2000, 400),(0,0,0))
#    draw = ImageDraw.Draw(mask)
    mask = np.zeros([400,2000,3], np.uint8)
    start = [100, 250]
    for char in text:
#        draw.text((start[0], start[1]), char, (cls[char],cls[char],cls[char]), font=font)
        mask[start[0]+font.getoffset(char)[1]:start[0]+font.getsize(char)[1]+font.getoffset(char)[1],start[1]+font.getoffset(char)[0]:start[1]+font.getoffset(char)[0]+font.getsize(char)[0],:] = cls[char]
        start[1] += font.getsize(char)[0]+font.getoffset(char)[0]
    img, mask = crop(img, mask, [128, 256])
    t = time.time()
    while True:
        pos0 = np.random.rand()*1.5 - .75
        pos1 = np.random.rand()*1.5 - .75
        scale0 = np.random.rand()*.5 + .25
        scale1 = np.random.rand()*.5 + .25
        if abs(pos0)+abs(scale0) > 1.25 or abs(pos1)+abs(scale1) > 1.25:
            continue
        img, mask = spherical(img, mask, pos=[pos0,pos1], scale=[scale0, scale1])
        break
    img, mask = crop(img, mask, [128, 256], border=10)
    img, mask = proj(img, mask, 0.35)
    img, mask = crop(img, mask, [48, 192], center=True)
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
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if im.shape[0] > im.shape[1]:
            c = (im.shape[0]-im.shape[1]) / 2
            im = im[c:c+im.shape[1],:,:]
        else:
            c = (im.shape[1]-im.shape[0]) / 2
            im = im[:,c:c+im.shape[0],:]
        im = cv2.resize(im, (size[1],size[0]))
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


    import random
    import string
    import cPickle as pickle
    font_dir = '/home/zw/dataset/fonts/fonts/'
    img_dir = '/home/zw/dataset/mscoco/'
    font_d = os.listdir(font_dir)
    img_d = os.listdir(img_dir)
    cls = {}
    chars = string.letters+string.digits
    for (i,c) in enumerate(chars):
        cls[c] = i+1
    count = 0
    while count < 50000:
        t = time.time()
        if random.random() > 0.5:
            while True:
                bg_img = cv2.imread(img_dir+random.choice(img_d))
                if bg_img == None:
                    continue
                bg_size = bg_img.shape
                if bg_size[0] < 384 or bg_size[1] < 384 or len(bg_size) != 3:
                    continue
                bg_img = crop_img(bg_img, [384,384])
                break
        else:
            maker = Maker(str(random.randint(0,15)), [384,384])
            bg_img = maker.generate()
#            bg_img = np.ones([384,384,3], np.uint8)
#            bg_img[:,:,0] *= 85*random.randint(0,3)
#            bg_img[:,:,1] *= 85*random.randint(0,3)
#            bg_img[:,:,2] *= 85*random.randint(0,3)
        if bg_img.shape != (384,384,3):
            continue
        bg_avg = bg_img[:192,:192,:].mean(axis=0).mean(axis=0)
        anno = np.zeros_like(bg_img)

        for ii in range(8):
            for jj in range(2):
                text = ''.join([random.choice(chars) for i in range(random.randint(5,12))])
                font = random.choice(font_d)
                image, mask = get_font(os.path.join(font_dir,font), text, cls)
                while True:
                    fg_img = cv2.imread(img_dir+random.choice(img_d))
                    if fg_img == None:
                        continue
                    fg_size = fg_img.shape
                    if fg_size[0] < 97 or fg_size[1] < 193 or len(fg_size) != 3:
                        continue
                    fg_img = cv2.cvtColor(fg_img, cv2.COLOR_BGR2RGB)
                    xstart = np.random.randint(0, fg_size[0]-96)
                    ystart = np.random.randint(0, fg_size[1]-192)
                    fg_img = fg_img[xstart:xstart+48,ystart:ystart+192,:].astype(np.float)
                    if fg_img.shape != (48,192,3):
                        continue
                    fg_avg = fg_img.mean(axis=0).mean(axis=0)
                    if abs(np.sum(fg_avg-bg_avg)) < 200:
                        continue
                    break
                enhance = (fg_avg>bg_avg) * 255
                fg_img[:,:,0] = fg_img[:,:,0]*0.75+enhance[0]*0.25
                fg_img[:,:,1] = fg_img[:,:,1]*0.75+enhance[1]*0.25
                fg_img[:,:,2] = fg_img[:,:,2]*0.75+enhance[2]*0.25
                image = image.astype(np.float)/255
                bg_img[48*ii:48*ii+48,192*jj:192*jj+192,:] = bg_img[48*ii:48*ii+48,192*jj:192*jj+192,:]*(1-image)+fg_img*image
                anno[48*ii:48*ii+48,192*jj:192*jj+192,:] = mask
        
        img = Image.fromarray(bg_img.astype(np.uint8))
        img.save('/home/zw/dataset/scene_text/%d.jpg'%count, quality=100)
        anno = anno[:,:,0].astype(np.uint8)
        anno = Image.fromarray(anno.astype(np.uint8))
        anno.save('/home/zw/dataset/scene_text/%d.png'%count, quality=100)
        print count 
        count += 1

if __name__ == '__main__':
    test()
