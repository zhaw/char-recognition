import mxnet as mx
import numpy as np

val_rec = mx.io.ImageRecordIter(
        path_imgrec = '/home/zw/Downloads/font_pics/font_pics_val.rec',
        mean_img    = '/home/zw/Downloads/font_pics/mean.nd',
        data_shape  = (1, 64, 64),
        batch_size  = 128,
        rand_crop   = False,
        rand_mirror = False
    )

model = mx.model.FeedForward.load('model', 5, mx.gpu())
pred = model.predict(val_rec)
pred = np.argmax(pred, axis=1)

truth = np.ones_like(pred)

with open('/home/zw/Downloads/font_pics/font_pics_val.lst') as f:
    lines = f.readlines()

idx = 0
for line in lines:
    _, c, _ = line.split('\t')
    truth[idx] = int(c)
    idx += 1


