import numpy as np
import tensorflow as tf
import random
import constants

"""Functions used in calculation of FGSM and random perturbations"""  
    
def _get_single_perturb_dir(model_builder, answer_num, feature_num):
  """
  Calculates the gradient of loss with respect to a singular scalar feature embedding. Used to calculate FGSM perturbation.

  Args:
    model_builder: The model in use
    answer_num: Integer representing answer to be perturbed.
    feature_num: Integer for feature for which to calculate gradient with respect to. Should be a number between 20-39, corresponding to embedding dimension.

  Returns:
    Gradient of loss with respect to singular feature embedding.
  """
  features = model_builder.normalized_features_evaluated
  direction = 0
  layer_size = int(constants._HIDDEN_LAYER_DIMS[0]) 
  for num in range(layer_size):
    grad = model_builder.grad_variable_pair_evaluated[0][feature_num][num] 
    weight = model_builder.grad_variable_pair_evaluated[1][feature_num][num]
    feature = features[answer_num][feature_num] 
    if feature != 0:
      direction += grad * (weight/feature) #mathmatically feature value has no impact here since in coming steps we take the sign of the calculation but was included for completeness
    else:
      direction = 0 

  return direction

def _get_perturb_dir(model_builder, answer_num):
  """
  Calculates gradient of loss with respect to features for entire answer. Used to calculate FGSM perturbation.

  Args:
    model_builder: The model in use.
    answer_num: Integer representing answer to be perturbed.

  Returns:
    Array of length 40 where the final 20 entries correspond to gradient of loss wrt to features that will be applied as noise to answer embeddings. First 20 entries correspond to noise applied to question embedding and thus are filled with 0s.
  """
  direction = []
  for num in range(20): 
    direction.append(0) #corresponds to question embedding which we do not want to perturb

  for num in range(20, 40): 
    direction.append(_get_single_perturb_dir(model_builder, answer_num, num))

  return direction

#apply FGSM equation and return array of perturbed input
def get_perturbed_input(model_builder, answer_num, perturb_amount):
  """
  Calculates noise to be added to input. Used to calculate FGSM perturbation.

  Args:
    model_builder: The model in use
    answer_num: Integer representing answer to be perturbed.
    perturb_amount: Float representing amount of perturbation.
  
  Returns:
    Array corresponding to noise to be added to input. Array has shape (_FULL_EMBEDDING, _BATCH_SIZE*_LIST_SIZE). 

  """
  noise = [[0]*constants._FULL_EMBEDDING]*constants._LIST_SIZE 

  if(model_builder.random_noise):
    direction = produce_random_noise() 
  else:
    direction = np.sign(_get_perturb_dir(model_builder, answer_num))
    
  scaled_direction = np.multiply(model_builder.perturb_amount, direction) 
  noise[answer_num] = scaled_direction.tolist()

  return noise

def produce_random_noise():
  """
  Generates random noise for which to compare against performance of FGSM generated noise.

  Returns:
    Array of length 40 where final 20 entries are noise applied to the answer embedding. First 20 entries are 0 as to not perturb question embedding.
  """
  random_noise = []
  for i in range(20): 
    random_noise.append(0) #corresponds to question embedding which we do not want to perturb
  for i in range(20): 
    num = np.sign(random.uniform(-1,1))
    random_noise.append(num)

  return random_noise

def calculate_grad_var_pair(model_builder):
  """
  Calculates the gradient of logits wrt weights in first dense layer of model. Used to calculate FGSM perturbation.

  Args:
    model_builder: The model in use.

  Returns:
    Tuple where first entry corresponds to gradients and second entry corresponds to weights.
  """
  model_builder.logits_tensor = model_builder.logits_tensor[model_builder.answer_number][0]
  model_builder.logits_tensor = tf.reshape(model_builder.logits_tensor, [1,1])
  
  graph_col = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES) #get grad of logits wrt to weights
  _gradient = model_builder.optimizer.compute_gradients(
      loss=model_builder.logits_tensor, var_list=graph_col)
  gradient = None 
  for x in _gradient:
    if 'group_score/dense/kernel:0' in x[1].name: #filter out everything except for weights/gradients in first dense layer 
      gradient = x

  return gradient 
