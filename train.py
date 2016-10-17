import mxnet as mx
import os
import numpy as np
import random
import cv2
import symbol

from PIL import Image

batch_size = 8 
path = '/home/zw/dataset/scene_text/'
imgout = mx.nd.zeros([batch_size,3,384,384], mx.gpu())
anno = mx.nd.zeros([batch_size,384,384], mx.gpu())

def get_data(batch_size, imgout, anno):
    n = len(os.listdir(path))
    for i in range(batch_size):
        chose = np.random.randint(0, n//2)
        im = np.array(Image.open('%s%d.jpg'%(path, chose))).astype(np.float)
        im = np.swapaxes(im, 0, 2)
        im = np.swapaxes(im, 1, 2)
        im -= 128
        imgout[i] = im
        tmp = np.array(Image.open('%s%d.png'%(path, chose)))
        tmp[tmp!=0] = 1
        anno[i] = tmp

def ce_loss(label, pred):
    pred = np.swapaxes(pred, 0, 1)
    pred = np.reshape(pred, [pred.shape[0], np.prod(pred.shape[1:])])
    pred = np.swapaxes(pred, 0, 1)
    label = np.reshape(label, [np.prod(label.shape)])
    prob = pred[np.arange(label.shape[0]), np.int64(label)]
    return -np.log(prob).sum() / label.shape[0]

net = symbol.symbol(2)

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
pretrained = mx.nd.load('imagenet-0005.params')
for name in arg_names:
    if 'arg:'+name in pretrained:
        pretrained['arg:'+name].copyto(arg_dict[name])
    elif name != 'data' and name != 'softmax_label':
        initializer(name, arg_dict[name])
net = net.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, aux_states=aux_dict, grad_req='write')

optimizer = mx.optimizer.SGD(learning_rate=1e-1, wd=1e-4, momentum=0.9)
optim_states = []
for i, var in enumerate(net.grad_dict):
    if var != 'data' and var != 'softmax_label':
        optim_states.append(optimizer.create_state(i, net.arg_dict[var]))
    else:
        optim_states.append([])


acc = 0
for batch in range(50000):
    if batch % 1000 == 0:
        mx.nd.save('args.nd', net.arg_dict)
        optimizer.lr /= 2
    get_data(batch_size, imgout, anno)
    net.forward(is_train=True)
    net.backward()
    for i, var in enumerate(net.grad_dict):
        if var != 'data' and var != 'softmax_label':
            optimizer.update(i, net.arg_dict[var], net.grad_dict[var], optim_states[i])
    if batch % 10 == 0:
        print (np.argmax(net.outputs[0].asnumpy(), axis=1)==net.arg_dict['softmax_label'].asnumpy()).mean(),
        print (np.argmax(net.outputs[0].asnumpy(), axis=1)!=0).mean(),
        print ce_loss(net.arg_dict['softmax_label'].asnumpy(), net.outputs[0].asnumpy())
