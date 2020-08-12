import sys
from visualizations import *
from model_builder import *

_TEST_DATA_PATH = "test.tfrecords"


if __name__ == "__main__":
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #Turn off console warnings
  tf.compat.v1.disable_eager_execution()

  model_builder = ModelBuilder()
  ranker = model_builder.run_training()

  init_variables(model_builder, ranker, _TEST_DATA_PATH, 0, .05, 1, True)
  
  save_ranking_bar_graph(model_builder, sys.argv)