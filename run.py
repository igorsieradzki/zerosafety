
from src.train import TrainPipeline

import tensorflow as tf
from pprint import PrettyPrinter


flags = tf.app.flags
flags.DEFINE_string("model", None, "model directory to load")
flags.DEFINE_integer("board_width", 6, "board width")
flags.DEFINE_integer("board_height", 6, "board heright")
flags.DEFINE_integer("n_in_row", 4, "how many stones in arow to win")
flags.DEFINE_float("lr", 2e-3, "learing rate for adam optimizer")
flags.DEFINE_integer("playouts", 400, "number of MCTS playouts")
flags.DEFINE_integer("batch_size", 512, "batch size for trainig policy/value net")
flags.DEFINE_integer("train_steps", 5, "number of train steps for polic/value net")
flags.DEFINE_integer("check_freq", 50, "after how many iteratons to evaluate")
flags.DEFINE_integer("iters", 1500, "number of training iterations")

FLAGS = flags.FLAGS

pp = PrettyPrinter()

def main(_):

    pp.pprint(flags.FLAGS.__flags)

    training_pipeline = TrainPipeline(init_model=FLAGS.model,
                                      board_width=FLAGS.board_width,
                                      board_height=FLAGS.board_height,
                                      n_in_row=FLAGS.n_in_row,
                                      learning_rate=FLAGS.lr,
                                      n_playouts=FLAGS.playouts,
                                      batch_size=FLAGS.batch_size,
                                      train_steps=FLAGS.train_steps,
                                      check_freq=FLAGS.check_freq,
                                      n_iters=FLAGS.iters)

    training_pipeline.run()


if __name__ == '__main__':
    tf.app.run()