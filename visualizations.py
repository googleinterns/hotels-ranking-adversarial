import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import constants
from model_eval import *
import constants

"""Visualizations for Model"""

_GREEN = '#3CAEA3'
_BLUE = '#20639B'
_RED = '#ED553B'
_YELLOW = '#FCE205'


def display_ranking_bar_graph(model_builder, argv):
    """
    Saves and displays bar graph comparing answer rankings for
    nonperturbed input, fgsm perturbed input and randomly perturbed input

    Args:
        model_builder: The model in use.
        argv: Command line input to be used for directory name.
    """
    barWidth = 0.25
    plt.figure(figsize=(22, 8))
    plt.rcParams.update({'font.size': 22})

    bars1 = model_builder.ranking_array
    bars2 = model_builder.perturbed_ranking_array
    bars3 = model_builder.random_ranking_array

    # Set position of bar on X axis
    r1 = np.arange(len(bars1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    barlist1 = plt.bar(
        r1,
        bars1,
        color=_GREEN,
        width=barWidth,
        label='Original Ranking')
    barlist2 = plt.bar(
        r2,
        bars2,
        color=_GREEN,
        width=barWidth,
        label='Perturbed FGSM Ranking')
    barlist3 = plt.bar(
        r3,
        bars3,
        color=_GREEN,
        width=barWidth,
        label='Perturbed Random Ranking')

    barlist1[model_builder.reference_number].set_color(_YELLOW)
    barlist2[model_builder.reference_number].set_color(_YELLOW)
    barlist3[model_builder.reference_number].set_color(_YELLOW)

    barlist2[model_builder.answer_number].set_color(_BLUE)
    barlist3[model_builder.answer_number].set_color(_RED)

    plt.xlabel('Answer ID', fontweight='bold')
    # Add xticks on the middle of the group bars
    plt.xticks([r + barWidth for r in range(len(bars1))],
               np.arange(1, len(bars1) + 1))

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

    if len(argv) > 1:
        plt.savefig(argv[1] + "/plot.png")
    else:
        plt.savefig("plot.png")
    plt.show()


def display_embedding_graph(model_builder, argv):
    '''
    Saves and displays graph of word embeddings, comparin embeddings of
    nonperturbed, fgsm perturbed and randomly perturbed input.

    Args:
        model_builder: The model in use.
        argv: Command line input to be used for directory name.
    '''
    barWidth = 0.25
    plt.figure(figsize=(22, 8))
    plt.rcParams.update({'font.size': 22})

    bars1 = model_builder.embedded_features_evaluated[
        model_builder.answer_number][constants._EMBEDDING_DIMENSION:]
    bars2 = model_builder.fgsm_noise_input
    bars3 = model_builder.random_noise_input

    # Set position of bar on X axis
    r1 = np.arange(1, len(bars1) + 1)
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]

    barlist1 = plt.bar(
        r1,
        bars1,
        color=_GREEN,
        width=barWidth,
        edgecolor='white',
        label='Original Embedding')
    barlist2 = plt.bar(
        r2,
        bars2,
        color=_BLUE,
        width=barWidth,
        edgecolor='white',
        label='Embedding with FGSM Noise')
    barlist3 = plt.bar(
        r3,
        bars3,
        color=_RED,
        width=barWidth,
        edgecolor='white',
        label='Embedding with Random Noise')

    # Add xticks on the middle of the group bars
    plt.xlabel('Embedding Dimension', fontweight='bold')
    plt.xticks([r + barWidth + 1 for r in range(len(bars1))],
               np.arange(1, len(bars1) + 1))

    plt.ylabel('Embedding Value', fontweight='bold')
    plt.title('Answer Embeddings', fontweight='bold')

    
    plt.legend()

    if len(argv) > 1:
        plt.savefig(argv[1] + "/embedding.png")
    else:
        plt.savefig("embedding.png")
    plt.show()
    


def display_perturbation_vs_epsilon(model_builder, ranker, answer_num, ref_num, epsilons):
    '''
    Calculates the difference between unperturbed input rank and fgsm/randomly 
    perturbed input over a variety of epsilon values.

    Args:
        model_builder: The model in use.
        ranker: Ranking object.
        answer_num: The answer that is perturbed.
        ref_num: The reference answer for the perturbed answer.
        epsilons: Array containing the different values for epsilon.

    Returns:
        amount_of_perturbation_fgsm: Array containing amount of 
        perturbation for FGSM perturbed input.
        amount_of_perturbation_random: Array containing amount of 
        perturbation for randomly perturbed input.

    '''
    amount_of_perturbation_fgsm = []
    amount_of_perturbation_random = []
    

    init_variables(model_builder, ranker, constants._TEST_DATA_PATH,
                   answer_num, epsilons[0], ref_num, True)
    for perturb in epsilons:
        init_variables(
            model_builder,
            ranker,
            constants._TEST_DATA_PATH,
            answer_num,
            perturb,
            ref_num,
            False)
        change_in_rank_fgsm = abs(
            model_builder.ranking_array[answer_num] -
            model_builder.perturbed_ranking_array[answer_num])
        amount_of_perturbation_fgsm.append(change_in_rank_fgsm)

        change_in_rank_random = abs(
            model_builder.ranking_array[answer_num] -
            model_builder.random_ranking_array[answer_num])
        amount_of_perturbation_random.append(change_in_rank_random)
    return amount_of_perturbation_fgsm, amount_of_perturbation_random


def perturbation_vs_epsilon_graph(model_builder, ranker, answer_num, ref_num, argv):
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
    #range of epsilons can be modified as desired to change
    #number of bars or increment size
    epsilons = np.arange(.1, .5, .1)
    amount_of_perturbation_fgsm, amount_of_perturbation_random = perturbation_vs_epsilon(
        model_builder, ranker, answer_num, ref_num, epsilons)

    barWidth = 0.25
    plt.figure(figsize=(22, 8))
    plt.rcParams.update({'font.size': 22})

    # Set position of bar on X axis
    r1 = np.arange(1, len(amount_of_perturbation_fgsm) + 1)
    r2 = [x + barWidth for x in r1]

    barlist1 = plt.bar(
        r1,
        amount_of_perturbation_fgsm,
        color='#3CAEA3',
        width=barWidth,
        edgecolor='white',
        label='Perturbation with FGSM Noise')
    barlist2 = plt.bar(
        r2,
        amount_of_perturbation_random,
        color='#20639B',
        width=barWidth,
        edgecolor='white',
        label='Perturbation with Random Noise')

    # Add xticks on the middle of the group bars
    plt.xlabel('Epsilon Value', fontweight='bold')
    plt.xticks(
        [r for r in range(1, len(amount_of_perturbation_fgsm) + 1)], (str(num)[:3] for num in epsilons))

    plt.ylabel('Difference in Ranking Score', fontweight='bold')
    plt.title('Amount of Perturbation vs Epsilon Value', fontweight='bold')

    plt.legend()

    if len(argv) > 1:
        plt.savefig(argv[1] + "/perturbation_v_epsilon.png")
    else:
        plt.savefig("perturbation_v_epsilon.png")
    plt.show()
  
