import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import shutil
import time
import sys
import numpy as np

from dlsrl.data_utils import get_data
from dlsrl.train_utils import get_batches, evaluate, shuffle_stack_pad, make_feed_dict, count_zeros_from_end
from dlsrl.model_v2 import Model
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
        model = Model(params, embeddings, label_dict.size())

        # eval  # TODO test - do not use sess.run
        x1, x2, y = shuffle_stack_pad(train_data, params.batch_size)

        x1_b = x1[:params.batch_size]  # word_ids
        x2_b = x2[:params.batch_size].astype(np.float32)  # predicate_ids

        lengths = [len(row) - count_zeros_from_end(row) for row in x1_b]
        print('lengths')
        print(lengths)
        max_seq_len = np.max(lengths)
        x1_b_max = x1_b[:, :max_seq_len]
        x2_b_max = x2_b[:, :max_seq_len]

        embedded = model.embed(x1_b_max)  # the whole dataset does not fit on GPU memory
        print(embedded)
        print(embedded.shape)

        mask = np.clip(x1_b, 0, 1)
        print(mask)
        print(mask.shape)
        encoded = model.encode_with_lstm(embedded, x2_b_max, mask)
        print(encoded)
        print(encoded.shape)



        sys.stdout.flush()
        raise SystemExit('Debugging')

        for epoch in range(params.max_epochs):

            # save checkpoint from which to load model
            ckpt_p = local_job_p / "epoch_{}.ckpt".format(epoch)
            ckpt_saver.save(sess, str(ckpt_p))
            print('Saved checkpoint.')

            # evaluate
            evaluate(dev_data, model, summary_writer, sess, epoch, global_step)
            x1, x2, y = shuffle_stack_pad(train_data, params.batch_size)
            epoch_start = time.time()

            for x1_b, x2_b, y_b in get_batches(x1, x2, y, params.batch_size):
                feed_dict = make_feed_dict(x1_b, x2_b, y_b, model, params.keep_prob)

                # tensorboard
                if global_step % config.Eval.summary_interval == 0:
                    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.NO_TRACE)
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

                # train
                loss, _ = sess.run([model.nonzero_mean_loss, model.update], feed_dict=feed_dict)
                global_step += 1

        sess.close()
        summary_writer.flush()
        summary_writer.close()

        #  move events file to shared drive
        events_p = list(local_job_p.glob('*events*'))[0]
        dst = config.RemoteDirs.runs / params.param_name / params.job_name
        if not dst.exists():
            dst.mkdir(parents=True)
        shutil.move(str(events_p), str(dst))


