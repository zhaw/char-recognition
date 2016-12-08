import mxnet as mx
import os
import symbol
import numpy as np
import string

from skimage import io, transform
from matplotlib import pyplot as plt
from PIL import Image, ImageFont, ImageDraw, ImageFilter
from PATH import *

pallete = []
colors = []
for i in range(4):
    for j in range(4):
        for k in range(4):
            pallete += [i*85, j*85, k*85]
            colors.append((i*85, j*85, k*85))

def make_legend(chars):
    font_path = os.path.join(FONTPATH, 'EricaOne-Regular.ttf')
    font = ImageFont.truetype(font_path, 60)
    img = Image.new("RGB", (512, 512), (0,0,0))
    draw = ImageDraw.Draw(img)
    for i in range(8):
        for j in range(8):
            if i*8+j == len(chars):
                break
            draw.text((30+j*60,i*60), chars[i*8+j], colors[i*8+j], font=font)
    img = np.array(img)
    return img

def crop_img(im, size):
    im = io.imread(im)
    if im.shape[0]*size[0] > im.shape[1]*size[1]:
        c = (im.shape[0]-1.*im.shape[1]/size[0]*size[1]) / 2
        c = int(c)
        im = im[c:-(1+c),:,:]
    else:
        c = (im.shape[1]-1.*im.shape[0]/size[1]*size[0]) / 2
        c = int(c)
        im = im[:,c:-(1+c),:]
    im = transform.resize(im, size)
    im *= 255
    return im

def preprocess_img(im, size):
    if type(size) == int:
        size = (size, size)
    im = crop_img(im, size)
    im = im.astype(np.float32)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im[0,:] -= 128
    im[1,:] -= 128
    im[2,:] -= 128
    im = np.expand_dims(im, 0)
    return im


class Recognizer():

    def __init__(self, size):
        s0, s1 = size
        s0 = s0//32*32
        s1 = s1//32*32
        self.s0 = s0
        self.s1 = s1
        net = symbol.symbol()
        args = mx.nd.load('args_res37.nd')
        auxs = mx.nd.load('auxs_res37.nd')
        for k in args:
            args[k] = args[k].copyto(mx.gpu())
        for k in auxs:
            auxs[k] = auxs[k].copyto(mx.gpu())
        args['data'] = mx.nd.zeros([1,3,s1,s0], mx.gpu(0))
        args['softmax_label'] = mx.nd.zeros([1,s1,s0], mx.gpu(0))
        self.executor = net.bind(ctx=mx.gpu(0), args=args, aux_states=auxs)

    def detect(self, img):
        self.executor.arg_dict['data'][:] = preprocess_img(img, (self.s0, self.s1))
        self.executor.forward()
        out = (self.executor.outputs[0].asnumpy()[0,:,:,:])
        return out
        out[0,:,:] /= 10
        maxpred = np.max(out, axis=0)
        cls = np.argmax(out, axis=0)
        maxpred[cls==0] = 0
        cls = Image.fromarray(cls.astype(np.uint8))
        cls.putpalette(pallete)
        cls.show()
        out = np.expand_dims(out, 2)
        out = np.tile(out, [1,1,3])
        plt.subplot(121)
        plt.imshow(crop_img(img, (self.s0, self.s1))/255)
        plt.subplot(122)
        plt.imshow(cls)
        plt.show()
#        io.imsave('x'+img.split('/')[-1], crop_img(img, (self.s0, self.s1)))
#        cls.save('x'+img.split('/')[-1].replace('jpg','png'))

class Recognizer2():

    def __init__(self, thres):
        self.ds = []
        for i in [96,128,192,256,384,512]:
            self.ds.append(Recognizer([i,i]))
        self.thres = thres
        self.legend = make_legend(' '+string.letters+string.digits)

    def detect(self, img):
        plt.figure(num=None, figsize=(18,8), dpi=80)
        plt.subplot(131)
        plt.imshow(crop_img(img, (512,512)).astype(np.uint8))
        result = np.zeros([63,512,512])
        for i in range(6):
#            plt.subplot(2,3,i+2)
            out = self.ds[i].detect(img)
            out = np.swapaxes(out, 0, 2)
            out = transform.resize(out, (512,512))
            out = np.swapaxes(out, 0, 2)
            result = np.maximum(result, out)
#            plt.imshow((out*255).astype(np.uint8))
        result[0] /= self.thres
        plt.subplot(132)
        maxpred = np.argmax(result, axis=0)
        cls = Image.fromarray(maxpred.astype(np.uint8))
        cls.putpalette(pallete)
        plt.imshow(cls)
        plt.subplot(133)
        plt.imshow(self.legend)
        fig = plt.gcf()
        def onclick(event):
            x = round(event.xdata)
            y = round(event.ydata)
            print (' '+string.letters+string.digits)[maxpred[y,x]]
        cid = fig.canvas.mpl_connect('motion_notify_event', onclick)
        plt.show()



class Detector():

    def __init__(self, size):
        s0, s1 = size
        s0 = s0//32*32
        s1 = s1//32*32
        self.s0 = s0
        self.s1 = s1
        net = symbol.symbol()
        args = mx.nd.load('args_res.nd')
        auxs = mx.nd.load('auxs_res.nd')
        args['data'] = mx.nd.zeros([1,3,s1,s0], mx.gpu())
        args['softmax_label'] = mx.nd.zeros([1,s1,s0], mx.gpu())
        self.executor = net.bind(ctx=mx.gpu(), args=args, aux_states=auxs)

    def detect(self, img):
        self.executor.arg_dict['data'][:] = preprocess_img(img, (self.s0, self.s1))
        self.executor.forward()
        out = (self.executor.outputs[0].asnumpy()[0,1,:,:])
        return out
        plt.subplot(121)
        plt.imshow(crop_img(img, (self.s0, self.s1)).astype(np.uint8))
        plt.subplot(122)
        plt.imshow((out*255).astype(np.uint8))
        plt.show()
#        io.imsave(img.split('/')[-1], crop_img(img, (self.s0, self.s1)))
#        io.imsave(img.split('/')[-1].replace('jpg','png'), (out*255).astype(np.uint8))

class Detector2():

    def __init__(self):
        self.ds = []
        for i in [128,192,256,384,512]:
            self.ds.append(Detector([i,i]))

    def detect(self, img):
        plt.figure(num=None, figsize=(20,10), dpi=80)
        plt.subplot(121)
        plt.imshow(crop_img(img, (512,512)).astype(np.uint8))
        result = np.zeros([1024,1024])
        for i in range(5):
#            plt.subplot(2,3,i+2)
            out = self.ds[i].detect(img)
            result = np.maximum(result, transform.resize(out, (1024,1024)))
#            plt.imshow((out*255).astype(np.uint8))
        plt.subplot(122)
        plt.imshow(result)
        plt.show()
#        io.imsave(img.split('/')[-1], crop_img(img, (self.s0, self.s1)))
#        io.imsave(img.split('/')[-1].replace('jpg','png'), (out*255).astype(np.uint8))

