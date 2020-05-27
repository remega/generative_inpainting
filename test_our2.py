import argparse

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng
import os
import glob
from inpaint_model import InpaintCAModel


# parser = argparse.ArgumentParser()
# parser.add_argument('--image', default='', type=str,
#                     help='The filename of image to be completed.')
# parser.add_argument('--mask', default='', type=str,
#                     help='The filename of mask, value 255 indicates mask.')
# parser.add_argument('--output', default='output.png', type=str,
#                     help='Where to write output.')
# parser.add_argument('--checkpoint_dir', default='', type=str,
#                     help='The directory of tensorflow checkpoint.')

inputdir = '../ValRemoval/SGITS'
outdir = './results/SGITS'
checkpoint_dir = './checkpoints/snap-0'

if not os.path.exists(outdir):
    os.makedirs(outdir)

if __name__ == "__main__":
    FLAGS = ng.Config('inpaint.yml')
    # ng.get_gpus(1)
    # args, unknown = parser.parse_known_args()
    model = InpaintCAModel()
    images = os.listdir(inputdir)
    images.sort()
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
      imin = tf.placeholder(tf.float32, (1, 256, 512, 3))
      output = model.build_server_graph(FLAGS, imin)
      output = (output + 1.) * 127.5
      output = tf.reverse(output, [-1])
      output = tf.saturate_cast(output, tf.uint8)
      # load pretrained model
      vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
      assign_ops = []
      for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable(checkpoint_dir, from_name)
        assign_ops.append(tf.assign(var, var_value))
        for x in images:
            curdir = os.path.join(inputdir, x)
            if os.path.isdir(curdir):
                image = cv2.imread(os.path.join(curdir, 'Ori.png'))
                mask = cv2.imread(os.path.join(curdir, 'mask.png'))
                assert image.shape == mask.shape

                h, w, _ = image.shape
                grid = 8
                image = image[:h//grid*grid, :w//grid*grid, :]
                mask = mask[:h//grid*grid, :w//grid*grid, :]
                print('Shape of image: {}'.format(image.shape))

                image = np.expand_dims(image, 0)
                mask = np.expand_dims(mask, 0)
                input_image = np.concatenate([image, mask], axis=2)
                input_image = tf.constant(input_image, dtype=tf.float32)

                sess.run(assign_ops)
                print('Model loaded.')
                result = sess.run(output, feed_dict={imin: input_image})

                cv2.imwrite(os.path.join(outdir, x + '.png'), result[0][:, :, ::-1])
