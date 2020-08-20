import sys
from absl import flags
from visualizations import *
from model_builder import *
import constants

flags.DEFINE_string("directory", "", "directory to save visualizations")
FLAGS = flags.FLAGS

if __name__ == "__main__":
    FLAGS(sys.argv)

    # Turn off console warning.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
    tf.compat.v1.disable_eager_execution()

    model_builder = ModelBuilder()
    ranker = model_builder.run_training()
    
    run(model_builder, ranker, constants._TEST_DATA_PATH, True)
    display_ranking_bar_graph(model_builder, FLAGS.directory)
    display_embedding_graph(model_builder, FLAGS.directory)

    # Optionally input parameters manually.
    '''
    #init_variables(model_builder=model_builder, ranker=ranker, 
    				path=constants._TEST_DATA_PATH, answer_num=1, 
    				perturb_amount=.01, reference_num=3, new_question=True)
    display_ranking_bar_graph(model_builder, FLAGS.directory)
    display_embedding_graph(model_builder, FLAGS.directory)
    '''

    # Additional optional visualization.
    # Note this may take a minute to run due to
    # data collection.
    '''
    display_perturbation_vs_epsilon_graph(model_builder=model_builder, 
    									ranker=ranker, answer_num=1, 
    									ref_num=2, directory=FLAGS.directory)
    '''
