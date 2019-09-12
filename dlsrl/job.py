import tensorflow as tf
from tensorflow.python.framework.errors_impl import DataLossError
import shutil
import time
import sys

from dlsrl.data_utils import get_data
from dlsrl.train_utils import get_batches, evaluate, shuffle_stack_pad, make_feed_dict
from dlsrl.model import Model
from dlsrl import config


class Params:

    def __init__(self, param2val):
        param2val = param2val.copy()

        self.param_name = param2val.pop('param_name')
        self.job_name = param2val.pop('job_name')

        self.param2val = param2val

    def __getattr__(self, name):
        if name in self.param2val:
            return self.param2val[name]
        else:
            raise AttributeError('No such attribute')

    def __str__(self):
        res = ''
        for k, v in sorted(self.param2val.items()):
            res += '{}={}\n'.format(k, v)
        return res


def main(param2val):

    # params
    params = Params(param2val)
    print(params)
    sys.stdout.flush()

    # make local folder for saving checkpoint + events files
    local_job_p = config.LocalDirs.runs / params.job_name
    if not local_job_p.exists():
        local_job_p.mkdir(parents=True)

    # data
    train_data, dev_data, word_dict, label_dict, embeddings = get_data(
        params, config.Eval.train_data_path, config.Eval.dev_data_path)
    sys.stdout.flush()

    # train loop
    global_step = 0
    global_start = time.time()
    tf_graph = tf.Graph()
    with tf_graph.as_default():

        # model
        model = Model(params, embeddings, label_dict.size(), tf_graph)
        sess = tf.Session(graph=tf_graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                log_device_placement=False))
        summary_writer = tf.summary.FileWriter(local_job_p, sess.graph)
        sess.run(tf.global_variables_initializer())
        ckpt_saver = tf.train.Saver(max_to_keep=params.max_epochs)

        # eval
        x1, x2, y = shuffle_stack_pad(train_data, params.train_batch_size)
        loss = sess.run(model.nonzero_mean_loss, feed_dict=make_feed_dict(x1, x2, y, model, keep_prob=1.0))
        sys.stdout.flush()

        for epoch in range(params.max_epochs):

            # save checkpoint from which to load model
            ckpt_p = local_job_p / "epoch_{}.ckpt".format(epoch)
            ckpt_saver.save(sess, str(ckpt_p))
            print('Saved checkpoint.')

            # evaluate
            evaluate(dev_data, model, summary_writer, sess, epoch, global_step)
            x1, x2, y = shuffle_stack_pad(train_data, params.train_batch_size)
            epoch_start = time.time()

            for x1_b, x2_b, y_b in get_batches(x1, x2, y, params.train_batch_size):
                feed_dict = make_feed_dict(x1_b, x2_b, y_b, model, params.keep_prob)
                if global_step % config.Eval.loss_interval == 0:

                    # tensorboard
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.NO_TRACE)
                    scalar_summaries = sess.run(model.scalar_summaries,
                                                feed_dict=feed_dict,
                                                options=run_options)
                    summary_writer.add_summary(scalar_summaries, global_step)

                    # print info
                    print("step {:>6} epoch {:>3}: loss={:1.3f}, epoch sec={:3.0f}, total hrs={:.1f}".format(
                        global_step,
                        epoch,
                        loss,
                        (time.time() - epoch_start),
                        (time.time() - global_start) / 3600))
                    sys.stdout.flush()

                loss, _ = sess.run([model.nonzero_mean_loss, model.update], feed_dict=feed_dict)
                global_step += 1


        sess.close()
        summary_writer.flush()
        summary_writer.close()

        # check data loss
        events_p = None
        for events_p in local_job_p.glob('*events*'):
            if is_data_loss(events_p):
                return RuntimeError('Detected data loss in events file. Did you close file writer?')

        #  move events file to shared drive
        dst = config.RemoteDirs.runs / params.param_name / params.job_name
        if not dst.exists():
            dst.mkdir(parents=True)
        shutil.move(str(events_p), str(dst))


def is_data_loss(events_p):
    try:
        list(tf.train.summary_iterator(str(events_p)))
    except DataLossError:
        return True
    else:
        return False