import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import constants
from model_eval import *

"""Visualizations for Model"""

_GREEN = '#3CAEA3'
_BLUE = '#20639B'
_RED = '#ED553B'
_YELLOW = '#FCE205'

_FONT_SIZE = 22
_BAR_WIDTH = 0.25
_FIG_SIZE = (22, 8)

_EPSILONS = np.arange(.1, .5, .1)

_RANKING_FILENAME = "ranking_bar_plot.png"
_EMBEDDING_FILENAME = "embedding_bar_plot.png"
_EPSILON_FILENAME = "perturbation_v_epsilon_plot.png"

def display_ranking_bar_graph(model_builder, argv):
    """
    Saves and displays bar graph comparing answer rankings for
    nonperturbed input, fgsm perturbed input and randomly perturbed input

    Args:
        model_builder: The model in use.
        argv: Command line input to be used for directory name.
    """
    plt.figure(figsize=_FIG_SIZE)
    plt.rcParams.update({'font.size': _FONT_SIZE})

    # Set position of bar on X axis
    ranking_range = np.arange(len(model_builder.ranking_array))
    fgsm_ranking_range = [x + _BAR_WIDTH for x in ranking_range]
    random_ranking_range = [x + _BAR_WIDTH for x in fgsm_ranking_range]

    unperturbed_barlist = plt.bar(
        ranking_range,
        model_builder.ranking_array,
        color=_GREEN,
        width=_BAR_WIDTH,
        label='Original Ranking')
    fgsm_barlist = plt.bar(
        fgsm_ranking_range,
        model_builder.perturbed_ranking_array,
        color=_GREEN,
        width=_BAR_WIDTH,
        label='Perturbed FGSM Ranking')
    random_barlist = plt.bar(
        random_ranking_range,
        model_builder.random_ranking_array,
        color=_GREEN,
        width=_BAR_WIDTH,
        label='Perturbed Random Ranking')

    unperturbed_barlist[model_builder.reference_number].set_color(_YELLOW)
    fgsm_barlist[model_builder.reference_number].set_color(_YELLOW)
    random_barlist[model_builder.reference_number].set_color(_YELLOW)
    fgsm_barlist[model_builder.answer_number].set_color(_BLUE)
    random_barlist[model_builder.answer_number].set_color(_RED)

    plt.xlabel('Answer ID', fontweight='bold')
    # Add xticks on the middle of the group bars
    plt.xticks([r + _BAR_WIDTH for r in range(len(model_builder.ranking_array))],
               np.arange(1, len(model_builder.ranking_array) + 1))
    plt.ylabel('Ranking Score', fontweight='bold')
    plt.title('Answer Rankings', fontweight='bold')

    # Create legend
    colors = [
        Line2D([0], [0], color=_GREEN, lw=8),
        Line2D([0], [0], color=_BLUE, lw=8),
        Line2D([0], [0], color=_RED, lw=8),
        Line2D([0], [0], color=_YELLOW, lw=8)]
    plt.legend(colors,
               ['Original Ranking',
                'Perturbed FGSM Ranking',
                'Perturbed Random Ranking',
                'Reference Answer Ranking'])

    plt.savefig(argv[1] + "/" + _RANKING_FILENAME)
    plt.show()


def display_embedding_graph(model_builder, argv):
    '''
    Saves and displays graph of word embeddings, comparing embeddings of
    nonperturbed, fgsm perturbed and randomly perturbed input.

    Args:
        model_builder: The model in use.
        argv: Command line input to be used for directory name.
    '''
    plt.figure(figsize=_FIG_SIZE)
    plt.rcParams.update({'font.size': _FONT_SIZE})

    unperturbed_bar = model_builder.embedded_features_evaluated[
        model_builder.answer_number][constants._EMBEDDING_DIMENSION:]
    fgsm_bar = model_builder.fgsm_embedding[
        model_builder.answer_number][constants._EMBEDDING_DIMENSION:]
    random_bar = model_builder.random_embedding[
        model_builder.answer_number][constants._EMBEDDING_DIMENSION:]

    # Set position of bar on X axis
    embedding_range = np.arange(1, len(unperturbed_bar) + 1)
    fgsm_embedding_range = [x + _BAR_WIDTH for x in embedding_range]
    random_embedding_range = [x + _BAR_WIDTH for x in fgsm_embedding_range]

    plt.bar(
        embedding_range,
        unperturbed_bar,
        color=_GREEN,
        width=_BAR_WIDTH,
        edgecolor='white',
        label='Original Embedding')
    plt.bar(
        fgsm_embedding_range,
        fgsm_bar,
        color=_BLUE,
        width=_BAR_WIDTH,
        edgecolor='white',
        label='Embedding with FGSM Noise')
    plt.bar(
        random_embedding_range,
        random_bar,
        color=_RED,
        width=_BAR_WIDTH,
        edgecolor='white',
        label='Embedding with Random Noise')

    # Add xticks on the middle of the group bars
    plt.xlabel('Embedding Dimension', fontweight='bold')
    plt.xticks([r + _BAR_WIDTH + 1 for r in range(len(unperturbed_bar))],
               np.arange(1, len(unperturbed_bar) + 1))

    plt.ylabel('Embedding Value', fontweight='bold')
    plt.title('Answer Embeddings', fontweight='bold')

    plt.legend()
    plt.savefig(argv[1] + "/" + _EMBEDDING_FILENAME)
    plt.show()


def perturbation_vs_epsilon(
        model_builder,
        ranker,
        answer_num,
        ref_num,
        epsilons):
    '''
    Calculates the difference between unperturbed input rank and 
    fgsm/randomly perturbed input over a variety of epsilon values.

    Args:
        model_builder: The model in use.
        ranker: Ranking object.
        answer_num: The answer that is perturbed.
        ref_num: The reference answer for the perturbed answer.
        epsilons: Array containing the different values for epsilon.

    Returns:
        returns a tuple (r1, r2) where r1 is an array containing the
        amount of perturbation for FGSM perturbed input and r2 is
        an array containing amount of perturbation for randomly
        perturbed input.
    '''
    amount_of_perturbation_fgsm = []
    amount_of_perturbation_random = []

    init_variables(
        model_builder=model_builder,
        ranker=ranker,
        path=constants._TEST_DATA_PATH,
        answer_num=answer_num,
        perturb_amount=epsilons[0],
        reference_num=ref_num,
        new_question=True)

    for num in range(1, len(epsilons)):
        change_in_rank_fgsm = abs(
            model_builder.ranking_array[answer_num] -
            model_builder.perturbed_ranking_array[answer_num])
        amount_of_perturbation_fgsm.append(change_in_rank_fgsm)

        change_in_rank_random = abs(
            model_builder.ranking_array[answer_num] -
            model_builder.random_ranking_array[answer_num])
        amount_of_perturbation_random.append(change_in_rank_random)
        init_variables(
            model_builder=model_builder,
            ranker=ranker,
            path=constants._TEST_DATA_PATH,
            answer_num=answer_num,
            perturb_amount=epsilons[num],
            reference_num=ref_num,
            new_question=False)
    change_in_rank_fgsm = abs(
        model_builder.ranking_array[answer_num] -
        model_builder.perturbed_ranking_array[answer_num])
    amount_of_perturbation_fgsm.append(change_in_rank_fgsm)

    change_in_rank_random = abs(
        model_builder.ranking_array[answer_num] -
        model_builder.random_ranking_array[answer_num])
    amount_of_perturbation_random.append(change_in_rank_random)
    return amount_of_perturbation_fgsm, amount_of_perturbation_random


def display_perturbation_vs_epsilon_graph(
        model_builder, ranker, answer_num, ref_num, argv):
    """
    Creates and saves bar graph showing difference in perturbation for both
    fgsm-perturbed and random-perturbed input for a variety of epsilon values.

    Args:
        model_builder: The model in use.
        ranker: Ranking object:
        answer_num: The answer that is perturbed.
        ref_num: The reference answer for the perturbed answer.
        argv: Command line input to be used for directory name.
    """
    # range of epsilons can be modified as desired to change
    # number of bars or increment size

    amount_of_perturbation_fgsm, amount_of_perturbation_random = perturbation_vs_epsilon(
        model_builder, ranker, answer_num, ref_num, _EPSILONS)

    plt.figure(figsize=_FIG_SIZE)
    plt.rcParams.update({'font.size': _FONT_SIZE})

    # Set position of bar on X axis
    fgsm_perturbation_range = np.arange(
        1, len(amount_of_perturbation_fgsm) + 1)
    random_perturbation_range = [
        x + _BAR_WIDTH for x in fgsm_perturbation_range]

    plt.bar(
        fgsm_perturbation_range,
        amount_of_perturbation_fgsm,
        color='#3CAEA3',
        width=_BAR_WIDTH,
        edgecolor='white',
        label='Perturbation with FGSM Noise')
    plt.bar(
        random_perturbation_range,
        amount_of_perturbation_random,
        color='#20639B',
        width=_BAR_WIDTH,
        edgecolor='white',
        label='Perturbation with Random Noise')

    # Add xticks on the middle of the group bars
    plt.xlabel('Epsilon Value', fontweight='bold')
    plt.xticks([r for r in range(1,
                                 len(amount_of_perturbation_fgsm) + 1)],
               ('{0:4.2f}'.format(num) for num in _EPSILONS))

    plt.ylabel('Difference in Ranking Score', fontweight='bold')
    plt.title('Amount of Perturbation vs Epsilon Value', fontweight='bold')

    plt.legend()
    plt.savefig(argv[1] + "/" + _EPSILON_FILENAME)
    plt.show()
