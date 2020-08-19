"""Methods used to print queries/answers"""

def _print_query(model_builder):
    """Prints singular question.

    Args:
      model_builder: The model in use.
    """
    query_tensor = model_builder.query_features_evaluated  
    values = query_tensor.values  

    for word in values:
        print(word.decode("utf-8"), end=' ')

def _create_answer_array(model_builder):
    """Creates and returns an array of unranked answers.

    Args:
      model_builder: The model in use

    Returns:
      An array containing the unranked answers for question selected.
    """
    doc_tensor = model_builder.answer_features_evaluated  
    values = doc_tensor.values 
    indices = doc_tensor.indices 

    answer_array = []
    single_answer = ''
    count = 0
    answer_number = 0

    for word in values:
        if(indices[count][1] == answer_number):
            single_answer += word.decode("utf-8") + ' '
        else:
            answer_array.append(single_answer)
            single_answer = word.decode("utf-8") + ' '
            answer_number += 1
        count += 1
    answer_array.append(single_answer)

    return answer_array

def create_ranked_answers_array(model_builder, ranking_array):
    """Combines an array of ranks with an array of features to create an array 
    relating each answer with its associated rank. Used for printing answers.

    Args:
      model_builder: The model in use.
      ranking_array: Array containing ranks of all answers.

    Returns:
      Array of tubles where each pair is of the format (answer, rank)
    """
    answer_array = _create_answer_array(model_builder)

    ranked_answers = []
    length = len(answer_array)

    for count in range(length):
        question_rank_pair = (answer_array[count], ranking_array[count])
        ranked_answers.append(question_rank_pair)
    # Answers with the highest score should come first
    ranked_answers.sort(key=lambda x: x[1], reverse=True)

    return ranked_answers

def print_ranked_answers(model_builder, ranking_array):
    """Prints question and associated answers according to ranking.

    Args:
      model_builder: The model in use.
      ranking_array: Array containing ranks of all answers.
    """

    ranked_answers = create_ranked_answers_array(model_builder, ranking_array)

    print('\n------------------------------------------------------------------')
    _print_query(model_builder)
    print('\n------------------------------------------------------------------')

    count = 1
    for answer_rank_pair in ranked_answers:
        print(count, ') ', end='')
        print(answer_rank_pair[0], '\n')
        count += 1

def print_settings(model_builder):
    '''
    Prints basic information about the perturbation, 
    including the question, answer and amount of perturbation.

    Args:
      model_builder: The model in use.
    '''
    print("Question: ", end='')
    _print_query(model_builder)
    print("")
    print("Answer Number: ", model_builder.answer_number)
    print(
        "Answer: ",
        _create_answer_array(model_builder)[
            model_builder.answer_number])
    print("Perturb amount: ", model_builder.perturb_amount)


