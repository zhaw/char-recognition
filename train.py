import mxnet as mx
import os
import numpy as np
import random
import cv2
import symbol

from PIL import Image
from PATH import *


batch_size = 1
n = len(os.listdir(DATAPATH))
imgout = mx.nd.zeros([batch_size,3,384,384], mx.gpu())
anno = mx.nd.zeros([batch_size,384,384], mx.gpu())


def get_image(i):
    chose = np.random.randint(0, n//3)
    im = np.array(Image.open(os.path.join(DATAPATH, '%d.jpg'%chose))).astype(np.float)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im -= 128
    tmp = np.array(Image.open(os.path.join(DATAPATH, '%d.png'%chose)))
    tmp[tmp!=0] = 1
    imgout[i] = im
    anno[i] = tmp
    return 0

def get_data(batch_size, imgout, anno):
    for i in range(batch_size):
        chose = np.random.randint(0, n//3)
        im = np.array(Image.open(os.path.join(DATAPATH, '%d.jpg'%chose))).astype(np.float)
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
        im -= 128
        tmp = np.array(Image.open(os.path.join(DATAPATH, '%d.png'%chose)))
#         tmp[tmp!=0] = 1
        imgout[i] = im
        anno[i] = tmp


def ce_loss(label, pred):
    pred = np.swapaxes(pred, 0, 1)
    pred = np.reshape(pred, [pred.shape[0], np.prod(pred.shape[1:])])
    pred = np.swapaxes(pred, 0, 1)
    label = np.reshape(label, [np.prod(label.shape)])
    prob = pred[np.arange(label.shape[0]), np.int64(label)]
    prob[prob<1e-10] = 1e-10
    return -np.log(prob).sum() / label.shape[0]

net = symbol.symbol()

initializer = mx.init.Normal(1e-3)
arg_shapes, output_shapes, aux_shapes = net.infer_shape(data=(batch_size,3,384,384), softmax_label=(batch_size,384,384))
arg_names = net.list_arguments()
arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
arg_dict['data'] = imgout
arg_dict['softmax_label'] = anno
aux_names = net.list_auxiliary_states()
aux_dict = dict(zip(aux_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in aux_shapes]))
grad_dict = {}
for k in arg_dict:
    if k != 'data' and k != 'softmax_label':
        grad_dict[k] = arg_dict[k].copyto(mx.gpu())
# pretrained = mx.nd.load('imagenet-0005.params')
for name in arg_names:
#     if 'arg:'+name in pretrained:
#         pretrained['arg:'+name].copyto(arg_dict[name])
    if name != 'data' and name != 'softmax_label':
        initializer(name, arg_dict[name])

net = net.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, aux_states=aux_dict, grad_req='write')

optimizer = mx.optimizer.SGD(learning_rate=1e-1, wd=1e-6, momentum=0.9)
optim_states = []
for i, var in enumerate(net.grad_dict):
    if var != 'data' and var != 'softmax_label':
        optim_states.append(optimizer.create_state(i, net.arg_dict[var]))
    else:
        optim_states.append([])


acc = 0
precision = 0
loss = 0
nonzeros = 0
for batch in range(2500000):
    if batch % 2500 == 0:
        mx.nd.save('args_res63.nd', net.arg_dict)
        mx.nd.save('auxs_res63.nd', net.aux_dict)
    if batch % 50000 == 0:
        optimizer.lr /= 2
    get_data(batch_size, imgout, anno)
#    pool.map(get_image, range(batch_size))
    net.forward(is_train=True)
    net.backward()
    for i, var in enumerate(net.grad_dict):
        if var != 'data' and var != 'softmax_label':
            optimizer.update(i, net.arg_dict[var], net.grad_dict[var], optim_states[i])
    precision += (np.argmax(net.outputs[0].asnumpy(), axis=1)==net.arg_dict['softmax_label'].asnumpy()).mean()
    nonzeros += (np.argmax(net.outputs[0].asnumpy(), axis=1)!=0).mean()
    loss += ce_loss(net.arg_dict['softmax_label'].asnumpy(), net.outputs[0].asnumpy())
    if batch % 10 == 0:
        print batch, precision/10, nonzeros/10, loss/10
        precision, nonzeros, loss = 0, 0, 0
