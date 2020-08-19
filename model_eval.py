import numpy as np
import constants
from print_answers import *
from fgsm_calculations import *
import copy

"""Functions to help with evaluation of model outside of model_builder"""

_TIMEOUT = 4

def run(model_builder, ranker, path, new_question):
    '''
    Saves information for perturbation and creates ranking arrays based on user input.

    Args:
        model_builder: The model in use.
        Ranker: Ranking object
        path: The path to test or training dataset.
        new_question: Boolean indicating if a new random question should be evaluated
    '''
    reset_flags(model_builder, new_question)
    
    model_builder.ranking_array = create_unperturbed_ranking_array(
        model_builder, ranker, path)
    print_ranked_answers(model_builder, model_builder.ranking_array)
    
    user_answer_num = int(input("Please select an answer to be perturbed: "))
    answer_num = convert_question_num(
        model_builder, user_answer_num)
    user_reference_num = int(input("Please select a reference answer: "))
    reference_num = convert_question_num(
        model_builder, user_reference_num)
    perturb_amount = float(
        input("Please select an amount of perturbation: "))

    init_variables(model_builder, ranker, path, answer_num, perturb_amount, reference_num, False)
    print_ranked_answers(model_builder, model_builder.perturbed_ranking_array)


def init_variables(
        model_builder,
        ranker,
        path,
        answer_num,
        perturb_amount,
        reference_num,
        new_question):
    """Saves information used for pertubation and creates ranking arrays.
    Args:
      model_builder: The model in use.
      ranker: Ranking object.
      path: The path to test or training dataset.
      answer_num: Answer to be perturbed.
      perturb_amount: Amount of perturbation applied (can be 0).
      reference_num: Answer to be used as reference for direction of perturbation.
      new_question: Boolean indicating if a new random question should be evaluated.
    """
    reset_flags(model_builder, new_question)
    model_builder.perturb_amount = perturb_amount
    model_builder.answer_number = answer_num
    model_builder.reference_number = reference_num

    model_builder.ranking_array = create_unperturbed_ranking_array(
        model_builder, ranker, path)
    model_builder.perturbed_ranking_array = create_fgsm_ranking_array(
        model_builder, ranker, path)
    model_builder.random_ranking_array = create_random_ranking_array(
        model_builder, ranker, path)

    remove_padding(model_builder)


def create_unperturbed_ranking_array(model_builder, ranker, path):
    '''
      Creates unperturbed ranking array.

      Args:
        model_builder: The model in use.
        ranker: ranker: Ranking object.
        path: The path to test or training dataset.

      Returns:
        Array containing ranks based on unperturbed input
    '''
    predictions = model_builder.custom_predict(
    False, ranker, input_fn=lambda: model_builder.predict_input_fn(path))

    return next(predictions)


def create_fgsm_ranking_array(model_builder, ranker, path):
    '''
    Creates fgsm perturbed ranking array.

    Args:
      model_builder: The model in use.
      ranker: ranker: Ranking object.
      path: The path to test or training dataset.

    Returns:
        Array containing ranks based on fgsm perturbed input.
    '''
    model_builder.perturb_amount = model_builder.perturb_amount * \
        get_fgsm_direction(model_builder)
    predictions = model_builder.custom_predict(
        True, ranker, input_fn=lambda: model_builder.predict_input_fn(path))

    return next(predictions)


def create_random_ranking_array(model_builder, ranker, path):
    '''
    Creates randomly perturbed ranking array. Note that ranking array is 
    repeatedly created until direction of perturbation matches the direction
    of fgsm perturbation. If after 5 tries result is still unsucessful,
    no random noise is generated.

    Args:
      model_builder: The model in use.
      ranker: ranker: Ranking object.
      path: The path to test or training dataset.

    Returns:
    Array containing ranks based on randomly perturbed input.
    '''
    model_builder.random_noise = True
    predictions = model_builder.custom_predict(
        True, ranker, input_fn=lambda: model_builder.predict_input_fn(path))

    rand_ranks = next(predictions)
    count = 0
    while correct_rand_noise_direction(model_builder, rand_ranks) == False:
        model_builder.random_noise_input = produce_random_noise()
        predictions = model_builder.custom_predict(
            True, ranker, input_fn=lambda: model_builder.predict_input_fn(path))
        rand_ranks = next(predictions)
        count += 1
        if count == _TIMEOUT:
            print("Timeout: No random noise created")
            model_builder.random_noise = False
            return model_builder.ranking_array
    model_builder.random_noise = False
    return rand_ranks

def correct_rand_noise_direction(model_builder, random_ranking_array):
    """
    Determines if randomly perturbed rank is perturbed in the same direction as
    fgsm perturbation.

    Args:
      model_builder: The model in use.
      random_ranking_array: Array of ranks based on randomly perturbed input.

    Returns:
      True if perturbation is in same direction as fgsm perturbation, False otherwise.
    """
    rank = model_builder.ranking_array[model_builder.answer_number]
    rand_rank = random_ranking_array[model_builder.answer_number]

    if get_fgsm_direction(model_builder) == 1:
      if rand_rank >= rank:
        return True
    else:
      if rand_rank <= rank:
        return True
    return False


def get_fgsm_direction(model_builder):
    """
    Determines whether to increase or decrease ranking so that perturbed
     question ranking approaches reference answer ranking.

    Args:
      model_builder: The model in use.
      ranking_array: Original nonperturbed array of answer rankings.

    Returns:
      1 or -1 where 1 indicates increasing the rank of the perturbed question
      and -1 indicates decreasing the rank of the perturbed question.
    """
    if model_builder.ranking_array[model_builder.answer_number] > model_builder.ranking_array[model_builder.reference_number]:
        #print("----------------Decreasing rank of question----------------")
        return -1
    else:
        #print("----------------Increasing rank of question----------------")
        return 1


def reset_flags(model_builder, new_question):
    """
    Reset flags in the model so a new question/answer pair can 
    be evaluated/perturbed.

    Args:
      model_builder: The model in use
      new_question: Boolean indicating if a new question
      should be evaluated
    """

    if new_question:
        model_builder.first_eval = True
    else:
        model_builder.first_eval = False
    


def convert_question_num(model_builder, answer_num):
    '''
    Converts the question number (based on sorted answers)
    that the user selects externally to interal question number

    Args:
      model_builder: The model in use.
      answer_num: Answer number selected externally by user

    Returns:
      Question number for internal use.
    '''
    answer_length = get_answer_size(model_builder)
    no_padding_ranking = model_builder.ranking_array[:answer_length]

    count = 0
    rank_w_question = []
    for rank in no_padding_ranking:
        rank_w_question.append((count, rank))
        count += 1
    rank_w_question.sort(key=lambda x: x[1], reverse=True)

    return rank_w_question[answer_num - 1][0]


def remove_padding(model_builder):
    '''
    Removes padding from ranking arrays

    Args:
      model_builder: The model in use.

    '''
    answer_length = get_answer_size(model_builder)
    model_builder.perturbed_ranking_array = model_builder.perturbed_ranking_array[
        :answer_length]
    model_builder.ranking_array = model_builder.ranking_array[:answer_length]
    model_builder.random_ranking_array = model_builder.random_ranking_array[:answer_length]


def get_answer_size(model_builder):
    """
    Determines amount of answers to remove answers from ranking arrays.
    """
    label = model_builder.labels_evaluated[0]
    # label of -1 corresponds to padding
    occurrences = np.count_nonzero(label == -1)
    return constants._LIST_SIZE - occurrences
