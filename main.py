import sys
from visualizations import *
from model_builder import *
import constants

if __name__ == "__main__":
	 # Turn off console warning
    tf.compat.v1.logging.set_verbosity(
        tf.compat.v1.logging.ERROR) 
    tf.compat.v1.disable_eager_execution()

    model_builder = ModelBuilder()
    ranker = model_builder.run_training()

    run(model_builder, ranker, constants._TEST_DATA_PATH, True)
    display_ranking_bar_graph(model_builder, sys.argv)
    display_embedding_graph(model_builder, sys.argv)

    #Optionally input parameters manually
    '''
    init_variables(model_builder, ranker, constants._TEST_DATA_PATH, 1, .1, 3, True)
    display_ranking_bar_graph(model_builder, sys.argv)
    display_embedding_graph(model_builder, sys.argv)
    '''

    #Additional optional visualizations
    #Note these may take a minute to run due to
    #data collection
    '''
    perturbation_vs_epsilon_graph(model_builder, ranker, 1, 2, sys.argv)
    '''
