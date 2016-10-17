import mxnet as mx
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def ConvFactory(data, num_filter, kernel, pad, stride=(1,1)):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad)
    bn = mx.symbol.BatchNorm(data=conv)
    act = mx.symbol.Activation(data=bn, act_type='relu')
    return act


def InceptionFactoryA(data, num_1x1, num_3x3red, num_3x3, num_d3x3red, num_d3x3, pool, proj):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), pad=(0,0))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), pad=(0,0))
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), pad=(0,0))
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool)
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), pad=(0,0))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd3x3, cproj])
    return concat


def InceptionFactoryB(data, num_3x3red, num_3x3, num_d3x3red, num_d3x3):
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), pad=(0,0))
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    # double 3x3 reduce + double 3x3
    cd3x3r = ConvFactory(data=data, num_filter=num_d3x3red, kernel=(1, 1), pad=(0,0))
    cd3x3 = ConvFactory(data=cd3x3r, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(1, 1))
    cd3x3 = ConvFactory(data=cd3x3, num_filter=num_d3x3, kernel=(3, 3), pad=(1, 1), stride=(2, 2))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # concat
    concat = mx.symbol.Concat(*[c3x3, cd3x3, pooling])
    return concat


def get_symbol(n_class):
    data = mx.sym.Variable('data')
    data = ConvFactory(data, 64, kernel=(5,5), pad=(2,2))
    data = mx.sym.Pooling(data=data, kernel=(3,3), stride=(2,2), pool_type='max')
    data = ConvFactory(data, 64, kernel=(1,1), pad=(0,0))
    data = ConvFactory(data, 96, kernel=(3,3), pad=(1,1))
    data = mx.sym.Pooling(data=data, kernel=(3,3), stride=(2,2), pool_type='max')
    data = InceptionFactoryA(data, 64, 64, 64, 64, 96, 'max', 32)
    data = InceptionFactoryB(data, 128, 160, 64, 96)
    data = InceptionFactoryA(data, 224, 64, 96, 128, 128, 'max', 128)
    data = InceptionFactoryB(data, 128, 192, 192, 256)
    data = InceptionFactoryA(data, 352, 192, 320, 160, 192, 'max', 128)
    data = mx.sym.Pooling(data=data, kernel=(4,4), stride=(1,1), pad=(0,0), pool_type='max')
    data = mx.sym.Flatten(data=data)
    data = mx.sym.FullyConnected(data=data, num_hidden=256)
    data = mx.sym.Activation(data=data, act_type='relu')
    data = mx.sym.FullyConnected(data=data, num_hidden=n_class)
    data = mx.sym.SoftmaxOutput(data=data, name='softmax')
    return data




batch_size = 128

train_rec = mx.io.ImageRecordIter(
        path_imgrec = '/home/zw/dataset/fonts/font_pics/font_train.rec',
        mean_img    = '/home/zw/dataset/fonts/font_pics/mean.nd',
        data_shape  = (1, 64, 64),
        batch_size  = batch_size,
        rand_crop   = False,
        rand_mirror = False
    )

val_rec = mx.io.ImageRecordIter(
        path_imgrec = '/home/zw/dataset/fonts/font_pics/font_test.rec',
        mean_img    = '/home/zw/dataset/fonts/font_pics/mean.nd',
        data_shape  = (1, 64, 64),
        batch_size  = batch_size,
        rand_crop   = False,
        rand_mirror = False
    )

model = mx.model.FeedForward(
        ctx         = mx.gpu(),
        symbol      = get_symbol(62),
        num_epoch   = 5,
        momentum    = 0.9,
        wd          = 1e-5,
        initializer = mx.init.Xavier(factor_type="in", magnitude=2.34),
        lr_scheduler= mx.lr_scheduler.FactorScheduler(step=500, factor=0.99)
    )

model.fit(X=train_rec, eval_data=val_rec, eval_metric=mx.metric.TopKAccuracy(top_k=2), batch_end_callback=mx.callback.Speedometer(batch_size, 50))
model.save('model')
