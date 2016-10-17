import time
import random
import os
import mxnet as mx
import numpy as np
np.set_printoptions(precision=2)
import cv2
import symbol

VGGPATH = './vgg19.params'

def postprocess_img(im, color_ref=None):
#     im = im[0]
    im[0,:] += 123.68
    im[1,:] += 116.779
    im[2,:] += 103.939
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    im[im<0] = 0
    im[im>255] = 255
    if color_ref != None:
        color_ref = cv2.cvtColor(color_ref, cv2.COLOR_RGB2HSV)
        im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2HSV)
        im[:,:,0] = color_ref[:,:,0]
        return cv2.cvtColor(im, cv2.COLOR_HSV2BGR)
    else:
        return cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2BGR)

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


def get_gram_executor(out_shapes, weights=[1,1,1,1,1]):
    gram_executors = []
    for i in range(len(weights)):
        shape = out_shapes[i]
        data = mx.sym.Variable('gram_data')
        flat = mx.sym.Reshape(data, shape=(int(shape[1]), int(np.prod(shape[2:]))))
        gram = mx.sym.FullyConnected(flat, flat, no_bias=True, num_hidden=shape[1]) # data shape: batchsize*n_in, weight shape: n_out*n_in
        normed = gram/np.prod(shape[1:])/shape[1]*weights[i]
        gram_executors.append(normed.bind(ctx=mx.gpu(), args={'gram_data':mx.nd.zeros(shape, mx.gpu())}, args_grad={'gram_data':mx.nd.zeros(shape, mx.gpu())}, grad_req='write'))
    return gram_executors


def get_tv_grad_executor(img, ctx, tv_weight):
    nchannel = img.shape[1]
    simg = mx.sym.Variable("img")
    skernel = mx.sym.Variable("kernel")
    channels = mx.sym.SliceChannel(simg, num_outputs=nchannel)
    out = mx.sym.Concat(*[
        mx.sym.Convolution(data=channels[i], weight=skernel,
                           num_filter=1,
                           kernel=(3, 3), pad=(1,1),
                           no_bias=True, stride=(1,1))
        for i in range(nchannel)])
    kernel = mx.nd.array(np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]])
                         .reshape((1, 1, 3, 3)),
                         ctx) / 8.0
    out = out * tv_weight
    return out.bind(ctx, args={"img": img,
                               "kernel": kernel})


def init_executor(m, batch_size, weights=[1,2,1,2], style_layers=['relu1_1','relu2_1','relu3_1','relu4_1'], content_layers=['relu4_2'], task='style'):
    size = 8*2**m
    initializer = mx.init.Xavier(factor_type='in', magnitude=2.34)
#     initializer = mx.init.Normal(1e-8)
    descriptor = symbol.descriptor_symbol(content_layers=content_layers, style_layers=style_layers, task=task)
    arg_shapes, output_shapes, aux_shapes = descriptor.infer_shape(data=(batch_size, 3, size, size))
    arg_names = descriptor.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
    grad_dict = {"data": arg_dict["data"].copyto(mx.gpu())}
    pretrained = mx.nd.load(VGGPATH)
    for name in arg_names:
        if name == "data":
            continue
        key = "arg:" + name
        if key in pretrained:
            pretrained[key].copyto(arg_dict[name])
    desc_executor = descriptor.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, grad_req='write')
    gram_executors = get_gram_executor(descriptor.infer_shape(data=(1,3,8*2**m,8*2**m))[1], weights=weights)
    generator = symbol.generator_symbol(m, task)
    if task == 'texture':
        z_shape = dict([('z_%d'%i, (batch_size,3,16*2**i,16*2**i)) for i in range(m)])
    else:
        z_shape = dict(
                [('zim_%d'%i, (batch_size,3,32*2**i,32*2**i)) for i in range(m)]
                )
    arg_shapes, output_shapes, aux_shapes = generator.infer_shape(**z_shape)
    arg_names = generator.list_arguments()
    arg_dict = dict(zip(arg_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in arg_shapes]))
    aux_names = generator.list_auxiliary_states()
    aux_dict = dict(zip(aux_names, [mx.nd.zeros(shape, ctx=mx.gpu()) for shape in aux_shapes]))
    grad_dict = {}
    for k in arg_dict:
        if k.startswith('block') or k.startswith('join'):
            grad_dict[k] = arg_dict[k].copyto(mx.gpu())
    for name in arg_names:
        if not name.startswith('z'):
            initializer(name, arg_dict[name])
    gene_executor = generator.bind(ctx=mx.gpu(), args=arg_dict, args_grad=grad_dict, aux_states=aux_dict, grad_req='write')
    return desc_executor, gram_executors, gene_executor


def train_texture(m, img, style_layer=['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'], weights=[1,1,1,1], max_epoch=100, lr=1e-8):
    size = 8*2**m
    batch_size = 1
    desc_executor, gram_executors, gene_executor = init_executor(m, batch_size, task='texture', weights=weights, style_layers=style_layer)
    optimizer = mx.optimizer.Adam(learning_rate=lr, wd=0e-5)
    optim_states = []
    for i, var in enumerate(gene_executor.grad_dict):
        if not var.startswith('z'):
            optim_states.append(optimizer.create_state(i, gene_executor.arg_dict[var]))
        else:
            optim_states.append([])
    im = preprocess_img('./%s'%img, size)
    target_grams = [mx.nd.zeros(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    im = mx.nd.array(im, mx.gpu())
    im.copyto(desc_executor.arg_dict['data'][:1])
    desc_executor.forward()
    gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
    for j in range(len(target_grams)):
        desc_executor.outputs[j][0:1].copyto(gram_executors[j].arg_dict['gram_data'])
        gram_executors[j].forward()
        target_grams[j][:] = gram_executors[j].outputs[0]
    old_loss = 0
    for epoch in range(max_epoch):
        if epoch % 50 == 49:
            for ii in range(batch_size):
                cv2.imwrite('out%s_%d_%d.jpg'%(img.split('.')[0], epoch, ii), postprocess_img(desc_executor.arg_dict['data'][ii].asnumpy()))#, color_ref=color_ref[selected[ii]]))
        if epoch in [249, 499, 999]:
            optimizer.lr /= 5
        if epoch % 2 == 0:
            print epoch,
        for j in range(m):
            for i in range(batch_size):
                gene_executor.arg_dict['z_%d'%j][i:i+1][:] = mx.random.uniform(-128,128,[1,3,16*2**j,16*2**j], mx.gpu())
        gene_executor.forward(is_train=True)
        gene_executor.outputs[0].copyto(desc_executor.arg_dict['data'])
        desc_executor.forward(is_train=True)
        loss = [0 for x in desc_executor.outputs]
        for i in range(batch_size):
            for j in range(len(style_layer)):
                desc_executor.outputs[j][i:i+1].copyto(gram_executors[j].arg_dict['gram_data'])
                gram_executors[j].forward(is_train=True)
                gram_diff[j] = gram_executors[j].outputs[0]-target_grams[j]
                gram_executors[j].backward(gram_diff[j])
                gram_grad[j][i:i+1] = gram_executors[j].grad_dict['gram_data'] / batch_size
                loss[j] += np.sum(np.square(gram_diff[j].asnumpy())) / batch_size
        improv = old_loss - sum(loss)
        old_loss = sum(loss)
        if epoch % 2 == 0:
            print 'loss', sum(loss), loss
        desc_executor.backward(gram_grad)
        gene_executor.backward(desc_executor.grad_dict['data'])
        for i, var in enumerate(gene_executor.grad_dict):
            if not var.startswith('z'):
                optimizer.update(i, gene_executor.arg_dict[var], gene_executor.grad_dict[var], optim_states[i])
        if epoch % 500 == 499:
            mx.nd.save('args%s_texture.nd'%img, gene_executor.arg_dict)
            mx.nd.save('auxs%s_texture.nd'%img, gene_executor.aux_dict)
    mx.nd.save('args%s_texture.nd'%img, gene_executor.arg_dict)
    mx.nd.save('auxs%s_texture.nd'%img, gene_executor.aux_dict)



def train_style(m, alpha, img, style_layer=['relu4_1'], content_layer=['relu4_2'], weights=[1], c_weights=[1], tv_weight=1e-4, max_epoch=100, lr=1e-8, early_stop=False, task='style'):
    size = 32*2**m
    batch_size = 1
    desc_executor, gram_executors, gene_executor = init_executor(m, batch_size, task='style', weights=weights, style_layers=style_layer, content_layers=content_layer)
    tv_grad_executor = get_tv_grad_executor(desc_executor.arg_dict['data'], mx.gpu(), tv_weight) 
    optimizer = mx.optimizer.Adam(learning_rate=lr, wd=1e-5)
    optim_states = []
    for i, var in enumerate(gene_executor.grad_dict):
        if not var.startswith('z'):
            optim_states.append(optimizer.create_state(i, gene_executor.arg_dict[var]))
        else:
            optim_states.append([])
    color_ref = [crop_img('data/%s'%file_name, 32*2**m) for file_name in os.listdir('data')]
    im_c = [[mx.nd.array(preprocess_img('data/%s'%file_name, 64*2**i), mx.gpu()) for file_name in os.listdir('data')] for i in range(m)]
    im = preprocess_img('./image%s.jpg'%img, size)
    target_grams = [mx.nd.zeros(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    for xy in range(2):
        for flx in range(2):
            for fly in range(2):
                imt = np.zeros_like(im)
                imt[:] = im
                if xy:
                    imt = np.swapaxes(imt, 2, 3)
                if flx:
                    imt = imt[:,:,::-1,:]
                if fly:
                    imt = imt[:,:,:,::-1]

                imt = mx.nd.array(imt, mx.gpu())
                imt.copyto(desc_executor.arg_dict['data'][:1])
                desc_executor.forward()
                gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
                gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
                for j in range(len(target_grams)):
                    desc_executor.outputs[j][0:1].copyto(gram_executors[j].arg_dict['gram_data'])
                    gram_executors[j].forward()
                    target_grams[j][:] += gram_executors[j].outputs[0]
    for j in range(len(target_grams)):
        target_grams[j][:] /= 8
    
    target_content = []
    for batch_idx in range(len(im_c[0])/batch_size):
        for i in range(batch_size):
            im_c[-1][batch_idx*batch_size+i].copyto(desc_executor.arg_dict['data'][i:i+1])
        desc_executor.forward()
        for i in range(batch_size):
            tmp = []
            for j in range(len(content_layer)):
                tmp.append(desc_executor.outputs[len(style_layer)+j][i:i+1].copyto(mx.gpu()))
            target_content.append(tmp)
    content_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[len(style_layer):]]

    stop = 300
    fine_tuned = 0
    c_best = 1e10
    stable = 0
    old_loss = 0
    improv = 0
    for epoch in range(max_epoch):
        if epoch % 50 == 49:
            for ii in range(batch_size):
                cv2.imwrite('out%s_%d_%d.jpg'%(img, epoch, ii), postprocess_img(desc_executor.arg_dict['data'][ii].asnumpy()))#, color_ref=color_ref[selected[ii]]))
#         if epoch in [200, 600, 1000, 1500, 2000, 2500, 3000, 3500]:
#             alpha /= 2
#             optimizer.lr *= 1.5
        if epoch % 500 == 499:
            optimizer.lr /= 2
        if epoch % 2 == 0:
            print epoch,
        stop -= 1
        if epoch % 2 == 0:
            print 'countdown', stop, 'stable', stable, 'improv', improv/batch_size,
        if stop == 0 and early_stop:
            if fine_tuned:
                break
            else:
                fine_tuned = 1
                optimizer.lr /= 10
                stop = 100
        if stable == 10:
            optimizer.lr *= 2
            stable = 0
        if task == 'style':
            selected = random.sample(range(len(im_c[0])), batch_size) 
            for j in range(m):
                for i in range(batch_size):
                    im_c[j][selected[i]].copyto(gene_executor.arg_dict['zim_%d'%j][i:i+1])
        else:
            for j in range(m):
                for i in range(batch_size):
                    gene_executor.arg_dict['zim_%d'%j][i:i+1][:] = mx.random.uniform(-128,128,[1,3,64*2**j,64*2**j], mx.gpu())
        gene_executor.forward(is_train=True)
        gene_executor.outputs[0].copyto(desc_executor.arg_dict['data'])
        tv_grad_executor.forward()
        desc_executor.forward(is_train=True)
        loss = [0 for x in desc_executor.outputs]
        for i in range(batch_size):
            for j in range(len(style_layer)):
                desc_executor.outputs[j][i:i+1].copyto(gram_executors[j].arg_dict['gram_data'])
                gram_executors[j].forward(is_train=True)
                gram_diff[j] = gram_executors[j].outputs[0]-target_grams[j]
                gram_executors[j].backward(gram_diff[j])
                gram_grad[j][i:i+1] = gram_executors[j].grad_dict['gram_data'] / batch_size
                loss[j] += np.sum(np.square(gram_diff[j].asnumpy())) / batch_size
            for j in range(len(content_layer)):
                layer_shape = desc_executor.outputs[len(style_layer)+j][i:i+1].shape
                layer_size = np.prod(layer_shape)
                loss[len(style_layer)+j] += c_weights[j]*alpha*np.sum(np.square((desc_executor.outputs[len(style_layer)+j][i:i+1]-target_content[selected[i]][j]).asnumpy()/np.sqrt(layer_size))) / batch_size
                content_grad[j][i:i+1] = c_weights[j]*alpha*(desc_executor.outputs[len(style_layer)+j][i:i+1]-target_content[selected[i]][j]) / layer_size / batch_size
        improv = old_loss - sum(loss)
        old_loss = sum(loss)
        if improv > 0 and improv*1e2 < sum(loss):
            stable += 1
        else:
            stable = 0
        if epoch % 2 == 0:
            print 'loss', sum(loss), loss
        if sum(loss)*1.1 < c_best:
            c_best = sum(loss)
            stop = 100 if fine_tuned else 300
        desc_executor.backward(gram_grad+content_grad)
        gene_executor.backward(desc_executor.grad_dict['data']+tv_grad_executor.outputs[0])
        for i, var in enumerate(gene_executor.grad_dict):
            if not var.startswith('z'):
                optimizer.update(i, gene_executor.arg_dict[var], gene_executor.grad_dict[var], optim_states[i])
        if epoch % 100 == 99:
            mx.nd.save('args%s_style.nd'%img, gene_executor.arg_dict)
            mx.nd.save('auxs%s_style.nd'%img, gene_executor.aux_dict)
    mx.nd.save('args%s_style.nd'%img, gene_executor.arg_dict)
    mx.nd.save('auxs%s_style.nd'%img, gene_executor.aux_dict)



def test2(m, alpha, img, weights=[1,1,4,1], tv_weight=1e-4, max_epoch=100, lr=1e-8):
    size = 32*2**m
    batch_size = 1 
    style_layer = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
    content_layer = ['relu4_2']
    style_layer = ['relu4_1']
    content_layer = ['relu4_2']
    desc_executor, gram_executors, gene_executor = init_executor(m, batch_size, task='style', style_layers=style_layer, content_layers=content_layer, weights=weights)
    tv_grad_executor = get_tv_grad_executor(desc_executor.arg_dict['data'], mx.gpu(), tv_weight) 
    im_c = [[mx.nd.array(preprocess_img('data/%s'%file_name, 64*2**i), mx.gpu()) for file_name in sorted(os.listdir('data'))] for i in range(m)]
    im = preprocess_img('./image%s.jpg'%img, size)
#     im = mx.nd.array(im, mx.gpu())
#     im.copyto(desc_executor.arg_dict['data'][:1])
#     desc_executor.forward()
    target_grams = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
    for xy in range(2):
        for flx in range(2):
            for fly in range(2):
                imt = np.zeros_like(im)
                imt[:] = im
                if xy:
                    imt = np.swapaxes(imt, 2, 3)
                if flx:
                    imt = imt[:,:,::-1,:]
                if fly:
                    imt = imt[:,:,:,::-1]

                imt = mx.nd.array(imt, mx.gpu())
                imt.copyto(desc_executor.arg_dict['data'][:1])
                desc_executor.forward()
                gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
                gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
                for j in range(len(target_grams)):
                    desc_executor.outputs[j][0:1].copyto(gram_executors[j].arg_dict['gram_data'])
                    gram_executors[j].forward()
                    target_grams[j][:] += gram_executors[j].outputs[0]
    for j in range(len(target_grams)):
        target_grams[j][:] /= 8
##
    im = mx.nd.array(im, mx.gpu())
    im.copyto(desc_executor.arg_dict['data'])
    desc_executor.forward()
    o = desc_executor.outputs[0].asnumpy()
    g = np.zeros([512*512,64,64])
    for x in range(64):
        for y in range(64):
            g[:,x,y] = np.outer(o[0,:,x,y],o[0,:,x,y]).flatten() / 512 / 512 
    from sklearn.cluster import KMeans as KM
    km = KM(4, verbose=3, max_iter=10, n_init=4, n_jobs=4)
    g = np.reshape(g, [g.shape[0], g.shape[1]*g.shape[2]])
    g = np.swapaxes(g, 0, 1)
    km.fit(g)
    target_gramss = []
    for i in range(4):
        target_gramss.append([mx.nd.array(np.reshape(km.cluster_centers_[i,:], [512,512]), mx.gpu())])
##


#     for j in range(len(target_grams)):
#         desc_executor.outputs[j][0:1].copyto(gram_executors[j].arg_dict['gram_data'])
#         gram_executors[j].forward()
#         gram_executors[j].outputs[0].copyto(target_grams[j])
    for iii in range(len(os.listdir('data'))):
        target_grams = target_gramss[iii%4]
        if os.path.exists('styled/%s.jpg'%(sorted(os.listdir('data'))[iii])):
            continue
        optimizer = mx.optimizer.Adam(learning_rate=lr, wd=0e-5)
        optim_states = optimizer.create_state(0, desc_executor.arg_dict['data'])
        target_content = []
        im_c[-1][iii].copyto(desc_executor.arg_dict['data'][0:1])
#     im_c[-1][0].copyto(desc_executor.arg_dict['data'][0:1])
        desc_executor.forward()
        tmp = []
        for j in range(len(content_layer)):
            tmp.append(desc_executor.outputs[len(style_layer)+j][0:1].copyto(mx.gpu()))
        target_content.append(tmp)
        content_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[len(style_layer):]]
        im_c[-1][iii].copyto(desc_executor.arg_dict['data'][0:1])
#         desc_executor.arg_dict['data'][0:1] = mx.random.uniform(-100,100,[1,3,512,512], mx.gpu())
        old_loss = 0
        for epoch in range(max_epoch):
            if epoch in [25, 50]:
                optimizer.lr *= 2
            if epoch in [250, 500, 1000]:
                optimizer.lr /= 2
            tv_grad_executor.forward(is_train=True)
            desc_executor.forward(is_train=True)
            loss = [0 for x in desc_executor.outputs]
            for i in range(batch_size):
                for j in range(len(style_layer)):
                    desc_executor.outputs[j][i:i+1].copyto(gram_executors[j].arg_dict['gram_data'])
                    gram_executors[j].forward(is_train=True)
                    gram_diff[j] = gram_executors[j].outputs[0]-target_grams[j]
                    gram_executors[j].backward(gram_diff[j])
                    gram_grad[j][i:i+1] = gram_executors[j].grad_dict['gram_data'] / batch_size
                    loss[j] += np.sum(np.square(gram_diff[j].asnumpy())) / batch_size
                for j in range(len(content_layer)):
                    layer_shape = desc_executor.outputs[len(style_layer)+j][i:i+1].shape
                    layer_size = np.prod(layer_shape)
                    loss[len(style_layer)+j] += alpha*np.sum(np.square((desc_executor.outputs[len(style_layer)+j][i:i+1]-target_content[0][j]).asnumpy()/np.sqrt(layer_size))) / batch_size
                    content_grad[j][i:i+1] = alpha*(desc_executor.outputs[len(style_layer)+j][i:i+1]-target_content[0][j]) / layer_size / batch_size
            improv = old_loss - sum(loss)
            old_loss = sum(loss)
            if epoch % 2 == 0:
                print epoch, 'loss', sum(loss), loss
            desc_executor.backward(gram_grad+content_grad)
            optimizer.update(0, desc_executor.arg_dict['data'], desc_executor.grad_dict['data']+tv_grad_executor.outputs[0], optim_states)
            result = desc_executor.arg_dict['data'].asnumpy()
            im = postprocess_img(result[0])
            cv2.imwrite('styled/%s'%sorted(os.listdir('data'))[iii], im)



def test(m, alpha, img, weights=[1,1,4,1], tv_weight=1e-4, max_epoch=100, lr=1e-8):
    size = 32*2**m
    batch_size = 1 
    style_layer = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']
    content_layer = ['relu4_2']
    style_layer = ['relu4_1']
    content_layer = ['relu4_2']
    desc_executor, gram_executors, gene_executor = init_executor(m, batch_size, task='style', style_layers=style_layer, content_layers=content_layer, weights=weights)
    tv_grad_executor = get_tv_grad_executor(desc_executor.arg_dict['data'], mx.gpu(), tv_weight) 
    im_c = [[mx.nd.array(preprocess_img('data/%s'%file_name, 64*2**i), mx.gpu()) for file_name in sorted(os.listdir('data'))] for i in range(m)]
    im = preprocess_img('./image%s.jpg'%img, size)
#     im = mx.nd.array(im, mx.gpu())
#     im.copyto(desc_executor.arg_dict['data'][:1])
#     desc_executor.forward()
    target_grams = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
    gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
    for xy in range(2):
        for flx in range(2):
            for fly in range(2):
                imt = np.zeros_like(im)
                imt[:] = im
                if xy:
                    imt = np.swapaxes(imt, 2, 3)
                if flx:
                    imt = imt[:,:,::-1,:]
                if fly:
                    imt = imt[:,:,:,::-1]

                imt = mx.nd.array(imt, mx.gpu())
                imt.copyto(desc_executor.arg_dict['data'][:1])
                desc_executor.forward()
                gram_diff = [mx.nd.empty(x.outputs[0].shape, mx.gpu()) for x in gram_executors]
                gram_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[:len(style_layer)]]
                for j in range(len(target_grams)):
                    desc_executor.outputs[j][0:1].copyto(gram_executors[j].arg_dict['gram_data'])
                    gram_executors[j].forward()
                    target_grams[j][:] += gram_executors[j].outputs[0]
    for j in range(len(target_grams)):
        target_grams[j][:] /= 8


#     for j in range(len(target_grams)):
#         desc_executor.outputs[j][0:1].copyto(gram_executors[j].arg_dict['gram_data'])
#         gram_executors[j].forward()
#         gram_executors[j].outputs[0].copyto(target_grams[j])
    for iii in range(len(os.listdir('data'))):
        if os.path.exists('styled/%s.jpg'%(sorted(os.listdir('data'))[iii])):
            continue
        optimizer = mx.optimizer.Adam(learning_rate=lr, wd=0e-5)
        optim_states = optimizer.create_state(0, desc_executor.arg_dict['data'])
        target_content = []
        im_c[-1][iii].copyto(desc_executor.arg_dict['data'][0:1])
#     im_c[-1][0].copyto(desc_executor.arg_dict['data'][0:1])
        desc_executor.forward()
        tmp = []
        for j in range(len(content_layer)):
            tmp.append(desc_executor.outputs[len(style_layer)+j][0:1].copyto(mx.gpu()))
        target_content.append(tmp)
        content_grad = [mx.nd.empty(x.shape, mx.gpu()) for x in desc_executor.outputs[len(style_layer):]]
        im_c[-1][iii].copyto(desc_executor.arg_dict['data'][0:1])
#         desc_executor.arg_dict['data'][0:1] = mx.random.uniform(-100,100,[1,3,512,512], mx.gpu())
        old_loss = 0
        for epoch in range(max_epoch):
            if epoch in [25, 50]:
                optimizer.lr *= 2
            if epoch in [250, 500, 1000]:
                optimizer.lr /= 2
            tv_grad_executor.forward(is_train=True)
            desc_executor.forward(is_train=True)
            loss = [0 for x in desc_executor.outputs]
            for i in range(batch_size):
                for j in range(len(style_layer)):
                    desc_executor.outputs[j][i:i+1].copyto(gram_executors[j].arg_dict['gram_data'])
                    gram_executors[j].forward(is_train=True)
                    gram_diff[j] = gram_executors[j].outputs[0]-target_grams[j]
                    gram_executors[j].backward(gram_diff[j])
                    gram_grad[j][i:i+1] = gram_executors[j].grad_dict['gram_data'] / batch_size
                    loss[j] += np.sum(np.square(gram_diff[j].asnumpy())) / batch_size
                for j in range(len(content_layer)):
                    layer_shape = desc_executor.outputs[len(style_layer)+j][i:i+1].shape
                    layer_size = np.prod(layer_shape)
                    loss[len(style_layer)+j] += alpha*np.sum(np.square((desc_executor.outputs[len(style_layer)+j][i:i+1]-target_content[0][j]).asnumpy()/np.sqrt(layer_size))) / batch_size
                    content_grad[j][i:i+1] = alpha*(desc_executor.outputs[len(style_layer)+j][i:i+1]-target_content[0][j]) / layer_size / batch_size
            improv = old_loss - sum(loss)
            old_loss = sum(loss)
            if epoch % 2 == 0:
                print epoch, 'loss', sum(loss), loss
            desc_executor.backward(gram_grad+content_grad)
            optimizer.update(0, desc_executor.arg_dict['data'], desc_executor.grad_dict['data']+tv_grad_executor.outputs[0], optim_states)
            result = desc_executor.arg_dict['data'].asnumpy()
            im = postprocess_img(result[0])
            cv2.imwrite('styled/%s'%sorted(os.listdir('data'))[iii], im)



