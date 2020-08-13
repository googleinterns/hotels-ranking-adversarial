import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import constants

"""Visualizations for Model"""

_GREEN = '#3CAEA3'
_BLUE = '#20639B'
_RED = '#ED553B'
_YELLOW = '#FCE205'


def save_ranking_bar_graph(model_builder, argv):
    """
    Saves and displays bar graph comparing answer rankings for
    nonperturbed input, fgsm perturbed input and randomly perturbed input
    """
    barWidth = 0.25
    plt.figure(figsize=(22, 8))
    plt.rcParams.update({'font.size': 16})

   
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

    plt.xlabel('Answer Number', fontweight='bold')
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


def display_embedding_graph(model_builder):
    barWidth = 0.25
    plt.figure(figsize=(22, 8))
    plt.rcParams.update({'font.size': 22})

    bars1 = 
        model_builder.embedded_features_evaluated[model_builder.answer_number][constants._EMBEDDING_DIMENSION:]
    bars2 = 
        model_builder.fgsm_noise_input[model_builder.answer_number][constants._EMBEDDING_DIMENSION:]
    bars3 = 
        model_builder.random_noise_input[model_builder.answer_number][constants._EMBEDDING_DIMENSION:]

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
    plt.xlabel('Answer Number', fontweight='bold')
    plt.xticks([r + barWidth + 1 for r in range(len(bars1))],
               np.arange(1, len(bars1) + 1))

    plt.ylabel('Embedding Value', fontweight='bold')
    plt.title('Answer Embeddings', fontweight='bold')

    # Create legend 
    plt.legend()
    plt.show()
