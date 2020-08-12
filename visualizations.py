import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

"""Visualizations for Model"""

def save_ranking_bar_graph(model_builder, argv):
  """
  Saves and displays bar graph comparing answer rankings for nonperturbed input, fgsm perturbed input and randomly perturbed input
  """
  barWidth = 0.25
  plt.figure(figsize=(22, 8))
  plt.rcParams.update({'font.size': 16})

  # set height of bar
  bars1 = model_builder.ranking_array 
  bars2 = model_builder.perturbed_ranking_array
  bars3 = model_builder.random_ranking_array
  
  # Set position of bar on X axis
  r1 = np.arange(len(bars1))
  r2 = [x + barWidth for x in r1]
  r3 = [x + barWidth for x in r2]
  
  # Make the plot
  barlist1 = plt.bar(r1, bars1, color='#3CAEA3', width=barWidth, label='Original Ranking')
  barlist2 = plt.bar(r2, bars2, color='#3CAEA3', width=barWidth, label='Perturbed FGSM Ranking')
  barlist3 = plt.bar(r3, bars3, color='#3CAEA3', width=barWidth, label='Perturbed Random Ranking')

  barlist1[model_builder.reference_number].set_color('#FCE205')
  barlist2[model_builder.reference_number].set_color('#FCE205')
  barlist3[model_builder.reference_number].set_color('#FCE205')

  barlist2[model_builder.answer_number].set_color('#20639B')
  barlist3[model_builder.answer_number].set_color('#ED553B')
  
  
  plt.xlabel('Answer Number', fontweight='bold')
  plt.xticks([r + barWidth for r in range(len(bars1))], np.arange(0, len(bars1))) # Add xticks on the middle of the group bars

  plt.ylabel('Ranking Score', fontweight='bold')
  plt.title('Answer Rankings', fontweight='bold')
  
  # Create legend & Show graphic
  colors =  [Line2D([0], [0], color='#3CAEA3', lw=8), Line2D([0], [0], color='#20639B', lw=8), Line2D([0], [0], color='#ED553B', lw=8), Line2D([0], [0], color='#FCE205', lw=8)]
  plt.legend(colors, ['Original Ranking', 'Perturbed FGSM Ranking', 'Perturbed Random Ranking', 'Reference Answer Ranking'])

  if len(argv) > 1:
    plt.savefig(argv[1] + "/plot.png")
  else:
    plt.savefig("plot.png")
  plt.show() 
