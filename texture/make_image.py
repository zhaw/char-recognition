import os
import time
import mxnet as mx
import numpy as np
import cv2
import symbol
import cPickle as pickle


def crop_img(im, size):
    im = cv2.imread(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if im.shape[0] > im.shape[1]:
        c = (im.shape[0]-im.shape[1]) / 2
        im = im[c:c+im.shape[1],:,:]
    else:
        c = (im.shape[1]-im.shape[0]) / 2
        im = im[:,c:c+im.shape[0],:]
    im = cv2.resize(im, (size,size))
    return im

def preprocess_img(im, size):
    im = crop_img(im, size)
    im = im.astype(np.float32)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im[0,:] -= 123.68
    im[1,:] -= 116.779
    im[2,:] -= 103.939
    im = np.expand_dims(im, 0)
    return im

def postprocess_img(im, color_ref=None):
    im = im[0]
    im[0,:] += 123.68
    im[1,:] += 116.779
    im[2,:] += 103.939
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    im[im<0] = 0
    im[im>255] = 255
    if color_ref != None:
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2HSV)
        im[:,:,0] = color_ref[:,:,0]
        print color_ref[:,:,0]
        return cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    else:
        return cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)

def make_texture(img, m, size):
    s0, s1 = size
    generator = symbol.generator_symbol(m, 'texture')
    args = mx.nd.load('args%s.jpg_texture.nd'%img)
    for i in range(m):
        args['z_%d'%i] = mx.nd.zeros([1,3,s0/16*2**i,s1/16*2**i], mx.gpu())
    gene_executor = generator.bind(ctx=mx.gpu(), args=args, aux_states=mx.nd.load('auxs%s.jpg_texture.nd'%img))
    for ii in range(1):
        for i in range(m):
            gene_executor.arg_dict['z_%d'%i][:] = np.random.uniform(-128,128,[1,3,s0/16*2**i,s1/16*2**i])
        gene_executor.forward()
        out = gene_executor.outputs[0].asnumpy()
        im = postprocess_img(out)
        cv2.imwrite('out%s_%d.jpg'%(img, ii), im)

def make_image(img, m, color_ref=None):
    generator = symbol.generator_symbol(m, 'style')
    args = mx.nd.load('args%s_style.nd'%img)
    for i in range(m):
        args['znoise_%d'%i] = mx.nd.zeros([1,1,16*2**i,16*2**i], mx.gpu())
        args['znoise_%d'%i][:] = np.random.uniform(-250,250,[1,1,16*2**i,16*2**i])
    auxs = mx.nd.load('auxs%s_style.nd'%img)
    with open('models/model%s.pkl'%img, 'w') as f:
        pickle.dump([args, auxs, generator], f)

    for i in range(m):
        args['znoise_%d'%i] = mx.nd.zeros([1,1,16*2**i,16*2**i], mx.gpu())
        args['zim_%d'%i] = mx.nd.zeros([1,3,16*2**i, 16*2**i], mx.gpu())
    gene_executor = generator.bind(ctx=mx.gpu(), args=args, aux_states=mx.nd.load('auxs%s_style.nd'%img))
    for test_im in os.listdir('test_pics'): 
        print test_im
        if color_ref:
            color_ref = cv2.cvtColor(crop_img('test_pics/%s'%test_im, 8*2**m), cv2.COLOR_RGB2HSV)
        for i in range(m):
            gene_executor.arg_dict['zim_%d'%i][:] = preprocess_img('test_pics/%s'%test_im, 16*2**i)
        for ii in range(4):
            t = time.clock()
            for i in range(m):
                gene_executor.arg_dict['znoise_%d'%i][:] = np.random.uniform(-150*ii,150*ii,[1,1,16*2**i,16*2**i])
            gene_executor.forward(is_train=True)
            out = gene_executor.outputs[0].asnumpy()
            im = postprocess_img(out, color_ref)
            cv2.imwrite('models/%s_%s_%d.jpg'%(test_im.split('.')[0], img, ii), im)


