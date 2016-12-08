import mxnet as mx
import os
import numpy as np
import random
import cv2
import symbol
import cPickle as pickle

from PIL import Image
from PATH import *


alpha = 1e0
batch_size = 1
ctx = mx.gpu(0)
n = len(os.listdir(DATAPATH))
n = 57 
imgout = mx.nd.zeros([batch_size,3,384,384], ctx)
anno = mx.nd.zeros([batch_size,63,384,384], ctx)
reg_anno = mx.nd.zeros([batch_size,2,384,384], ctx)
cls_grad = mx.nd.zeros([batch_size,63,384,384], ctx)
reg_grad = mx.nd.zeros([batch_size,2,384,384], ctx)


def get_image(i):
    chose = np.random.randint(0, n//3)
    im = np.array(Image.open(os.path.join(DATAPATH, '%d.jpg'%chose))).astype(np.float)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im -= 128
    tmp = np.array(Image.open(os.path.join(DATAPATH, '%d.png'%chose)))
#     tmp[tmp!=0] = 1
    for i in range(26):
        tmp[tmp==(27+i)] = 1+i
    imgout[i] = im
    anno[i] = tmp
    return 0

def get_data(batch_size, imgout, anno, reg_anno):
    for i in range(batch_size):
        chose = np.random.randint(0, n//3)
        im = np.array(Image.open(os.path.join(DATAPATH, '%d.jpg'%chose))).astype(np.float)
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
        im -= 128
        tmp = np.array(Image.open(os.path.join(DATAPATH, '%d.png'%chose)))
#         tmp[tmp!=0] = 1
        for j in range(26):
            tmp[tmp==(27+j)] = 1+j
        tmp_a = tmp.flatten()
        tmp = np.zeros([63,384*384])
        tmp[tmp_a,np.arange(384*384)] = 1
        tmp = np.reshape(tmp, [63,384,384])
        imgout[i] = im
        anno[i] = tmp
        with open(os.path.join(DATAPATH, '%d.pkl'%chose)) as f:
            d = pickle.load(f)
        d[0] = np.swapaxes(d[0], 0, 2)
        d[0] = np.swapaxes(d[0], 1, 2)
        reg_anno[i] = d[0]


net = symbol.reg_symbol(63)

initializer = mx.init.Normal(1e-3)
arg_shapes, output_shapes, aux_shapes = net.infer_shape(data=(batch_size,3,384,384)) 
arg_names = net.list_arguments()
arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=ctx) for shape in arg_shapes]))
arg_dict['data'] = imgout
arg_dict['softmax_label'] = anno
arg_dict['linear_regression_label'] = reg_anno
aux_names = net.list_auxiliary_states()
aux_dict = dict(zip(aux_names, [mx.nd.zeros(shape, ctx=ctx) for shape in aux_shapes]))
grad_dict = {}
for k in arg_dict:
    if k != 'data' and k != 'softmax_label' and k != 'linear_regression_label':
        grad_dict[k] = arg_dict[k].copyto(ctx)
pretrained_args = mx.nd.load('args_res37.nd')
pretrained_auxs = mx.nd.load('auxs_res37.nd')
for name in arg_names:
#     if 'arg:'+name in pretrained:
#         pretrained['arg:'+name].copyto(arg_dict[name])
    if name != 'data':
        if name in pretrained_args and not name.startswith('score'):
#             print name
#             print pretrained_args[name].shape, arg_dict[name].shape
            pretrained_args[name].copyto(arg_dict[name])
        else:
            initializer(name, arg_dict[name])
for name in aux_names:
    if name in pretrained_auxs:
        pretrained_auxs[name].copyto(aux_dict[name])
    else:
        initializer(name, aux_dict[name])

net = net.bind(ctx=ctx, args=arg_dict, args_grad=grad_dict, aux_states=aux_dict, grad_req='write')

optimizer = mx.optimizer.SGD(learning_rate=1e-8, wd=1e-6, momentum=0.9)
optim_states = []
for i, var in enumerate(net.grad_dict):
    if var != 'data':
        optim_states.append(optimizer.create_state(i, net.arg_dict[var]))
    else:
        optim_states.append([])


acc = 0
precision = 0
loss = 0
reg_loss = 0
nonzeros = 0
get_data(batch_size, imgout, anno, reg_anno)
for batch in range(1, 2500000):
    if batch % 25000 == 0:
        args_to_save = {}
        auxs_to_save = {}
        for k in net.arg_dict:
            args_to_save[k] = net.arg_dict[k].copyto(mx.cpu())
        for k in net.aux_dict:
            auxs_to_save[k] = net.aux_dict[k].copyto(mx.cpu())
        mx.nd.save('args_reg37.nd', args_to_save) # avoid device ordinal problem
        mx.nd.save('auxs_reg37.nd', auxs_to_save)
    if batch % 50000 == 0:
        optimizer.lr /= 2
#    pool.map(get_image, range(batch_size))
    net.forward(is_train=True)
    
    cls_grad = net.outputs[0] - anno
    cls_pred_np = net.outputs[0].asnumpy()
    cls_truth_np = anno.asnumpy()
#    print net.outputs[1].asnumpy()[0,0,50:60,50:60]
#    print reg_anno.asnumpy()[0,0,50:60,50:60]
    precision += np.mean(np.argmax(cls_pred_np, axis=1)==np.argmax(cls_truth_np, axis=1))
    reg_grad_np = (net.outputs[1] - reg_anno).asnumpy()
    anno_np = anno.asnumpy()
    for i in range(batch_size):
        reg_grad_np[i,:,anno_np[i,0,:,:]==1] = 0
#    print reg_grad_np[0,0,50:60,50:60]
#    print '\n'*3
    reg_grad[:] = alpha * reg_grad_np
    net.backward([cls_grad, reg_grad])
    get_data(batch_size, imgout, anno, reg_anno)
    for i, var in enumerate(net.grad_dict):
#        if var.startswith('score') or (var not in pretrained_args and var != 'data'):
        if var != 'data':
            optimizer.update(i, net.arg_dict[var], net.grad_dict[var], optim_states[i])
    loss += np.mean(np.square(cls_grad.asnumpy()))
    reg_loss += alpha * np.mean(np.square(reg_grad_np))
    if batch % 10 == 0:
        print batch, loss/10, reg_loss/10, precision/10
        precision, nonzeros, loss, reg_loss = 0, 0, 0, 0
