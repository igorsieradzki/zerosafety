
from src.train import TrainPipeline

import tensorflow as tf
from pprint import pprint
import os
from src import results_dir
from datetime import datetime


flags = tf.app.flags
flags.DEFINE_string("model", None, "model directory to load")
flags.DEFINE_integer("board_width", 6, "board width")
flags.DEFINE_integer("board_height", 6, "board heright")
flags.DEFINE_integer("n_in_row", 4, "how many stones in arow to win")
flags.DEFINE_float("lr", 2e-3, "learing rate for optimizer")
flags.DEFINE_integer("playouts", 400, "number of MCTS playouts")
flags.DEFINE_integer("batch_size", 512, "batch size for trainig policy/value net")
flags.DEFINE_integer("train_steps", 5, "number of train steps for polic/value net")
flags.DEFINE_integer("check_freq", 50, "after how many iteratons to evaluate")
flags.DEFINE_integer("iters", 1500, "number of training iterations")
flags.DEFINE_string("save_dir", None, "dir to save the model in")
flags.DEFINE_boolean("debug", False, "Debug mode")

FLAGS = flags.FLAGS

def main(_):

    pprint(flags.FLAGS.__flags)

    # create save dir for this run
    now = datetime.now()

    if FLAGS.save_dir is None:
        save_dir = os.path.join(results_dir, "zero_{day}_{m}_{h}:{min}".format(day=now.day,
                                                                          m=now.month,
                                                                          h=now.hour,
                                                                          min=now.minute))
    else:
        save_dir = os.path.join(results_dir, FLAGS.save_dir)

    if not os.path.exists(save_dir) and not FLAGS.debug:
        os.makedirs(save_dir)

        with open(os.path.join(save_dir, "params.txt"), 'w+') as file:
            pprint(flags.FLAGS.__flags, stream=file)

    training_pipeline = TrainPipeline(init_model=FLAGS.model,
                                      board_width=FLAGS.board_width,
                                      board_height=FLAGS.board_height,
                                      n_in_row=FLAGS.n_in_row,
                                      learning_rate=FLAGS.lr,
                                      n_playouts=FLAGS.playouts,
                                      batch_size=FLAGS.batch_size,
                                      train_steps=FLAGS.train_steps,
                                      check_freq=FLAGS.check_freq,
                                      n_iters=FLAGS.iters,
                                      save_dir=save_dir)

    training_pipeline.run()


if __name__ == '__main__':
    tf.app.run()