import numpy as np
import constants

"""Functions to help with evaluation of model outside of model_builder"""

def reset_flags(model_builder, new_question):
  """
  Reset flags in the model so a new question/answer pair can be evaluated/perturbed.

  Args:
    model_builder: The model in use
  """
  if new_question:
    model_builder.first_eval = True
  else:
    model_builder.first_eval = False
  model_builder.random_noise = False

def init_variables(model_builder, ranker, path, answer_num, perturb_amount, reference_num, new_question):
  """Saves information used for pertubation and creates ranking arrays.
  Args:

    path: The path to test or training dataset.
    answer_num: Answer to be perturbed.
    perturb_amount: Amount of perturbation applied (can be 0).
    reference_num: Answer to be used as reference for direction of perturbation.
    new_question: Boolean indicating if a new random question should be evaluated 
  """
  reset_flags(model_builder, new_question)
  model_builder.perturb_amount = perturb_amount 
  model_builder.answer_number = answer_num
  model_builder.reference_number = reference_num
  
  
  model_builder.perturbed_ranking_array, model_builder.ranking_array, model_builder.random_ranking_array = create_ranking_array(model_builder, ranker, path)
  
  answer_length = get_answer_size(model_builder) #get rid of padding in ranking arrays
  model_builder.perturbed_ranking_array = model_builder.perturbed_ranking_array[:answer_length] 
  model_builder.ranking_array = model_builder.ranking_array[:answer_length] 
  model_builder.random_ranking_array = model_builder.random_ranking_array[:answer_length] 

def create_ranking_array(model_builder, ranker, path):
  """
  Creates an array of ranks corresponding to question number for nonperturbed input, fgsm perturbed input and randomly perturbed input.
  
  Args:
    model_builder: The model in use.
    ranker:The ranker object used in predict function.
    path: The path to test or training dataset.

  Returns:
    Array of ranks for nonperturbed input, fgsm perturbed input and randomly perturbed input.
  """
  
  #for non-perturbed input
  predictions = model_builder.custom_predict(False, ranker, input_fn=lambda: model_builder.predict_input_fn(path))
  ranking_array = next(predictions)

  #for FGSM perturbed input
  model_builder.perturb_amount = model_builder.perturb_amount * get_fgsm_direction(model_builder, ranking_array)
  predictions = model_builder.custom_predict(True, ranker, input_fn=lambda: model_builder.predict_input_fn(path)) 
  perturbed_ranking_array = next(predictions)

  #for random noise perturbed input
  model_builder.random_noise = True
  predictions = model_builder.custom_predict(True, ranker, input_fn=lambda: model_builder.predict_input_fn(path))
  random_ranking_array = next(predictions)

  return perturbed_ranking_array, ranking_array, random_ranking_array


def get_fgsm_direction(model_builder, ranking_array):
  """
  Determines whether to increase or decrease ranking so that perturbed question ranking approaches reference answer ranking.

  Args:
    model_builder: The model in use.
    ranking_array: Original nonperturbed array of answer rankings.

  Returns:
    1 or -1 where 1 indicates increasing the rank of the perturbed question and -1 indicates decreasing the rank of the perturbed question.
  """
  if ranking_array[model_builder.answer_number] > ranking_array[model_builder.reference_number]:
    print("----------------Decreasing rank of question----------------")
    return -1
  else:
    print("----------------Increasing rank of question----------------")
    return 1

def get_answer_size(model_builder):
  """
  Determines amount of answers to remove answers from ranking arrays.
  """
  label = model_builder.labels_evaluated[0]
  occurrences = np.count_nonzero(label == -1) #label of -1 corresponds to padding
  return constants._LIST_SIZE - occurrences
