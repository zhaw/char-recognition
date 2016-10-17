import mxnet as mx

def conv(data, kernel, num_filter, stride, pad, name):
    data = mx.sym.Convolution(data=data, kernel=(kernel,kernel), num_filter=num_filter, stride=(stride,stride), pad=(pad,pad), name='%s_conv'%name)
    data = mx.sym.BatchNorm(data=data, name='%s_batchnorm'%name)
    data = mx.sym.Activation(data=data, act_type='relu')
    return data

def res_block(data, num_filter, name):
    data2 = conv(data, 3, num_filter, 1, 1, '%s_conv'%name)
    data2 = mx.sym.Convolution(data=data2, kernel=(3,3), num_filter=num_filter, stride=(1,1), pad=(1,1), name='%s_conv'%name)
    data2 = mx.sym.BatchNorm(data=data2, name='%s_batchnorm'%name)
    data = data + data2
    data = mx.sym.Activation(data=data, act_type='relu')
    return data

def block(data, num_filter, name):
    data_d = conv(data, 3, num_filter*2, 2, 1, '%s_downconv'%name) 
    data_d = res_block(data_d, num_filter*2, '%s_downres1'%name)
    data_d = res_block(data_d, num_filter*2, '%s_downres2'%name)
    data = res_block(data, num_filter, '%s_res1'%name)
    data = res_block(data, num_filter, '%s_res2'%name)
    data = res_block(data, num_filter, '%s_res3'%name)
    data_d = mx.sym.Deconvolution(data=data_d, kernel=(2,2), pad=(0,0), stride=(2,2), num_filter=num_filter, name='%s_updeconv'%name)
    data_d = mx.sym.BatchNorm(data=data_d, name='%s_deconvbn'%name)
    data = data + data_d
    return mx.sym.Activation(data=data, act_type='relu')


def symbol(n_class=63):
    data = mx.sym.Variable('data')
    data = conv(data, 7, 64, 2, 3, 'first') # 192
    data = block(data, 64, 'block1')
    data = block(data, 64, 'block2')
    data = block(data, 64, 'block3')
#    data = block(data, 64, 'block4')
#    data = block(data, 64, 'block5')
#    data = block(data, 64, 'block6')
    data = mx.sym.Deconvolution(data=data, kernel=(2,2), pad=(0,0), stride=(2,2), num_filter=64, name='last_deconv')
    data = mx.sym.BatchNorm(data=data, name='last_deconvbn')
    data = mx.sym.Activation(data=data, act_type='relu')
#    data = mx.sym.Dropout(data=data, p=0.5, name='dropout')
    data = mx.sym.Convolution(data=data, kernel=(1,1), num_filter=n_class, pad=(0,0), stride=(1,1), name='score')
    data = mx.sym.SoftmaxOutput(data=data, multi_output=True, name='softmax')
    return data

def symbol(n_class=63):
    data = mx.sym.Variable('data')
    data = conv(data, 7, 64, 2, 3, '1')
    data = conv(data, 3, 96, 1, 1, '2')
    data = conv(data, 3, 96, 1, 1, '3')
    data = mx.sym.Pooling(data, kernel=(2,2), pad=(0,0), stride=(2,2), pool_type='max')
    data = conv(data, 3, 128, 1, 1, '4')
    data = conv(data, 3, 128, 1, 1, '5')
    data = mx.sym.Pooling(data, kernel=(2,2), pad=(0,0), stride=(2,2), pool_type='max')
    data = conv(data, 3, 192, 1, 1, '6')
    data = conv(data, 3, 192, 1, 1, '7')
    data = mx.sym.Pooling(data, kernel=(2,2), pad=(0,0), stride=(2,2), pool_type='max')
    data = conv(data, 3, 256, 1, 1, '8')
    data = conv(data, 3, 256, 1, 1, '9')
    data = mx.sym.Deconvolution(data=data, kernel=(16,16), pad=(0,0), stride=(16,16), num_filter=64, name='last_deconv')
    data = mx.sym.BatchNorm(data=data, name='last_deconvbn')
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.Convolution(data=data, kernel=(1,1), num_filter=n_class, pad=(0,0), stride=(1,1), name='score')
    data = mx.sym.SoftmaxOutput(data=data, multi_output=True, name='softmax')
    return data


def get_imagenet_symbol():
    data = mx.sym.Variable('data')
    data = conv(data, 7, 64, 2, 3, '1')
    data = conv(data, 3, 96, 1, 1, '2')
    data = conv(data, 3, 96, 1, 1, '3')
    data = mx.sym.Pooling(data, kernel=(2,2), pad=(0,0), stride=(2,2), pool_type='max')
    data = conv(data, 3, 128, 1, 1, '4')
    data = conv(data, 3, 128, 1, 1, '5')
    data = mx.sym.Pooling(data, kernel=(2,2), pad=(0,0), stride=(2,2), pool_type='max')
    data = conv(data, 3, 192, 1, 1, '6')
    data = conv(data, 3, 192, 1, 1, '7')
    data = mx.sym.Pooling(data, kernel=(2,2), pad=(0,0), stride=(2,2), pool_type='max')
    data = conv(data, 3, 256, 1, 1, '8')
    data = conv(data, 3, 256, 1, 1, '9')
    data = mx.sym.Pooling(data, kernel=(7,7), pad=(0,0), stride=(7,7), pool_type='max')
    data = mx.sym.Flatten(data)
    data = mx.sym.FullyConnected(data=data, num_hidden=1000)
    data = mx.sym.SoftmaxOutput(data=data, name='softmax')
    return data

#def symbol(n_class=63):
#    data = mx.sym.Variable('data')
#    data = conv(data,7,64,2,3,'1')
#    data = conv(data,3,128,1,1,'2')
#    data = conv(data,3,128,1,1,'3')
#    data = conv(data,3,128,2,1,'4')
#    data = conv(data,3,256,1,1,'5')
#    data = conv(data,3,256,1,1,'6')
#    data = conv(data,3,256,2,1,'7')
#    data = conv(data,3,256,1,1,'8')
#    data = conv(data,3,256,1,1,'9')
#    data = mx.sym.Deconvolution(data=data, kernel=(2,2), pad=(0,0), stride=(2,2), num_filter=256)
#    data = mx.sym.BatchNorm(data=data)
#    data = mx.sym.Activation(data=data, act_type='relu')
#    data = mx.sym.Deconvolution(data=data, kernel=(2,2), pad=(0,0), stride=(2,2), num_filter=128)
#    data = mx.sym.BatchNorm(data=data)
#    data = mx.sym.Activation(data=data, act_type='relu')
#    data = mx.sym.Deconvolution(data=data, kernel=(2,2), pad=(0,0), stride=(2,2), num_filter=128)
#    data = mx.sym.BatchNorm(data=data)
#    data = mx.sym.Activation(data=data, act_type='relu')
#    data = mx.sym.Convolution(data=data, kernel=(1,1), num_filter=n_class, pad=(0,0), stride=(1,1))
#    data = mx.sym.SoftmaxOutput(data=data, multi_output=True, name='softmax')
#    return data
