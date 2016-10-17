import logging
import mxnet as mx
from symbol import get_imagenet_symbol

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def main():
    train_iter = mx.io.ImageRecordIter(
            path_imgrec = '/home/zw/dataset/ImageNet/train.rec',
            mean_r = 123.68,
            mean_g = 116.779,
            mean_b = 103.939,
            data_shape = (3, 224, 224),
            batch_size = 48,
            rand_crop = True,
            rand_mirror = True,
        )

    val_iter = mx.io.ImageRecordIter(
            path_imgrec = '/home/zw/dataset/ImageNet/val.rec',
            mean_r = 123.68,
            mean_g = 116.779,
            mean_b = 103.939,
            data_shape = (3, 224, 224),
            batch_size = 48,
        )

    model = mx.model.FeedForward(ctx=mx.gpu(), symbol=get_imagenet_symbol(), num_epoch=10000, learning_rate=1e-2, 
            momentum=0.9, wd=1e-8, initializer=mx.init.Normal(sigma=0.01), 
            lr_scheduler=mx.lr_scheduler.FactorScheduler(step=20000, factor=0.96))
    eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k=5)]

    model.fit(
            X = train_iter,
            eval_data = val_iter,
            eval_metric = eval_metric,
            batch_end_callback = mx.callback.Speedometer(128, 50),
            epoch_end_callback = mx.callback.do_checkpoint('imagenet')
        )

if __name__ == '__main__':
    main()
