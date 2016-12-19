import mxnet as mx
import os
import numpy as np
import random
import cv2
import symbol
import cPickle as pickle

from PIL import Image
from PATH import *
from pathos.multiprocessing import Pool
pool = Pool(12)


alpha = 1e-2
batch_size = 480
ctx = mx.gpu(1)
n = len(os.listdir(DATAPATH))
n = 1500000 
imgout = mx.nd.zeros([1,3,384,384], ctx)
anno = mx.nd.zeros([1,37,384,384], ctx)
anno_np = np.zeros([1,37,384,384])
reg_anno = mx.nd.zeros([1,74,384,384], ctx)
reg_anno_np = np.zeros([1,74,384,384])
cls_grad = mx.nd.zeros([1,37,384,384], ctx)
reg_grad = mx.nd.zeros([1,74,384,384], ctx)
reg_grad_np = np.zeros([1,74,384,384])


def get_image():
    result = []
    chose = np.random.randint(0, 1500000//3)
    im = np.array(Image.open(os.path.join(DATAPATH, '%d.jpg'%chose))).astype(np.float)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im -= 128
    tmp = np.array(Image.open(os.path.join(DATAPATH, '%d.png'%chose)))
    for j in range(26):
        tmp[tmp==(27+j)] = 1+j
    for j in range(10):
        tmp[tmp==(53+j)] = 27+j
    tmp_a = tmp.flatten()
    tmp = np.zeros([37,384*384])
    tmp[tmp_a,np.arange(384*384)] = 1
    tmp = np.reshape(tmp, [37,384,384])
    result.append(im)
    result.append(tmp)
    with open(os.path.join(DATAPATH, '%d.pkl'%chose)) as f:
        d = pickle.load(f)
    d[0] = np.swapaxes(d[0], 0, 2)
    d[0] = np.swapaxes(d[0], 1, 2)
    reg_anno_np = np.tile(d[0], [37,1,1])
    for j in range(37):
        reg_anno_np[j*2:j*2+2,tmp[j,:,:]==0] = 0
    result.append(reg_anno_np)
    return result

def get_data(batch_size, imgout, anno_np, reg_anno_np):
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
        for j in range(10):
            tmp[tmp==(53+j)] = 27+j
        tmp_a = tmp.flatten()
        tmp = np.zeros([37,384*384])
        tmp[tmp_a,np.arange(384*384)] = 1
        tmp = np.reshape(tmp, [37,384,384])
        imgout[i] = im
        anno[i] = tmp
        with open(os.path.join(DATAPATH, '%d.pkl'%chose)) as f:
            d = pickle.load(f)
        d[0] = np.swapaxes(d[0], 0, 2)
        d[0] = np.swapaxes(d[0], 1, 2)
        reg_anno_np[i] = np.tile(d[0], [37,1,1])


net = symbol.reg_symbol_simple(37)

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
pretrained_args = {} #mx.nd.load('args_res37.nd')
pretrained_auxs = {} #mx.nd.load('auxs_res37.nd')
for name in arg_names:
#     if 'arg:'+name in pretrained:
#         pretrained['arg:'+name].copyto(arg_dict[name])
    if name != 'data':
#         if name in pretrained_args and not name.startswith('score'):
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

optimizer = mx.optimizer.SGD(learning_rate=1e-4, wd=1e-6, momentum=0.9)
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
for batch in range(1, 2500000):
    if batch % 10 == 0:
        args_to_save = {}
        auxs_to_save = {}
        for k in net.arg_dict:
            args_to_save[k] = net.arg_dict[k].copyto(mx.cpu())
        for k in net.aux_dict:
            auxs_to_save[k] = net.aux_dict[k].copyto(mx.cpu())
        mx.nd.save('args_reg37ss.nd', args_to_save) # avoid device ordinal problem
        mx.nd.save('auxs_reg37ss.nd', auxs_to_save)
    if batch % 20 == 0:
        optimizer.lr /= 2
        anno[:] = anno_np
    data = [pool.apply_async(get_image, ()) for i in range(batch_size)]
#     get_data(batch_size, imgout, anno_np, reg_anno_np)
    for bb in range(batch_size):
        d = data[bb].get()
        imgout[0] = d[0]
        anno[0] = d[1]
        reg_anno[0] = d[2]
        net.forward(is_train=True)
        
        cls_grad = net.outputs[0] - anno
        cls_pred_np = net.outputs[0].asnumpy()
        cls_truth_np = anno.asnumpy()
#    print net.outputs[1].asnumpy()[0,0,50:60,50:60]
#    print reg_anno.asnumpy()[0,0,50:60,50:60]
        precision += np.mean(np.argmax(cls_pred_np, axis=1)==np.argmax(cls_truth_np, axis=1))
#         for i in range(batch_size):
#             for j in range(37):
#                 reg_grad_np[:] = net.outputs[1].asnumpy() - reg_anno_np
#                 reg_grad_np[i,j*2:j*2+2,anno_np[i,j,:,:]==0] = 0
#    print reg_grad_np[0,0,50:60,50:60]
#    print '\n'*3
        reg_grad[:] = alpha * (net.outputs[1]-reg_anno)
        net.backward([cls_grad, reg_grad])
        for i, var in enumerate(net.grad_dict):
#        if var.startswith('score') or (var not in pretrained_args and var != 'data'):
            if var != 'data':
                optimizer.update(i, net.arg_dict[var], net.grad_dict[var], optim_states[i])
        loss += np.mean(np.square(cls_grad.asnumpy()))
        reg_loss += alpha * np.mean(np.square(reg_grad.asnumpy()))
    print batch, loss/batch_size, reg_loss/batch_size, precision/batch_size
    precision, nonzeros, loss, reg_loss = 0, 0, 0, 0
