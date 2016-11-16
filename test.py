import mxnet as mx
import symbol
import cv2
import numpy as np

from skimage import io, transform
from matplotlib import pyplot as plt
from PIL import Image

pallete = []
for i in range(4):
    for j in range(4):
        for k in range(4):
            pallete += [i*85, j*85, k*85]

def crop_img(im, size):
    im = cv2.imread(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[0]*size[0] > im.shape[1]*size[1]:
        c = (im.shape[0]-1.*im.shape[1]/size[0]*size[1]) / 2
        c = int(c)
        im = im[c:-(1+c),:,:]
    else:
        c = (im.shape[1]-1.*im.shape[0]/size[1]*size[0]) / 2
        c = int(c)
        im = im[:,c:-(1+c),:]
    im = cv2.resize(im, size)
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
        args = mx.nd.load('args2.nd')
        auxs = mx.nd.load('auxs2.nd')
        args['data'] = mx.nd.zeros([1,3,s1,s0], mx.gpu())
        args['softmax_label'] = mx.nd.zeros([1,s1,s0], mx.gpu())
        self.executor = net.bind(ctx=mx.gpu(), args=args, aux_states=auxs)

    def detect(self, img):
        self.executor.arg_dict['data'][:] = preprocess_img(img, (self.s0, self.s1))
        self.executor.forward()
        out = (self.executor.outputs[0].asnumpy()[0,:,:,:])
        maxpred = np.max(out, axis=0)
        cls = np.argmax(out, axis=0)
        maxpred[cls==0] = 0
        cls = Image.fromarray(cls.astype(np.uint8))
        cls.putpalette(pallete)
#        cls.show()
        out = np.expand_dims(out, 2)
        out = np.tile(out, [1,1,3])
#        plt.subplot(121)
#        plt.imshow(crop_img(img, (self.s0, self.s1)))
#        plt.subplot(122)
#        plt.imshow(maxpred)
#        plt.show()
        io.imsave('x'+img.split('/')[-1], crop_img(img, (self.s0, self.s1)))
        cls.save('x'+img.split('/')[-1].replace('jpg','png'))


class D2():
    def __init__(self):
        size = [128,192,256,384,512,768]
        self.ds = []
        for s in size:
            self.ds.append(Detector([s,s]))

    def detect(self, img):
        for d in self.ds:
            d.detect(img)




class Detector():

    def __init__(self, size):
        s0, s1 = size
        s0 = s0//32*32
        s1 = s1//32*32
        self.s0 = s0
        self.s1 = s1
        net = symbol.symbol()
        args = mx.nd.load('args.nd')
        auxs = mx.nd.load('auxs.nd')
        args['data'] = mx.nd.zeros([1,3,s1,s0], mx.gpu())
        args['softmax_label'] = mx.nd.zeros([1,s1,s0], mx.gpu())
        self.executor = net.bind(ctx=mx.gpu(), args=args, aux_states=auxs)

    def detect(self, img):
        self.executor.arg_dict['data'][:] = preprocess_img(img, (self.s0, self.s1))
        self.executor.forward()
        out = (self.executor.outputs[0].asnumpy()[0,1,:,:])
        out = np.expand_dims(out, 2)
        out = np.tile(out, [1,1,3])
#        plt.subplot(121)
#        plt.imshow(crop_img(img, (self.s0, self.s1)))
#        plt.subplot(122)
#        plt.imshow(maxpred)
#        plt.show()
        io.imsave('result/'+img.split('/')[-1], crop_img(img, (self.s0, self.s1)))
        io.imsave('result/'+img.split('/')[-1].replace('.jpg','_%d.png'%self.s0), (out*255).astype(np.uint8))

