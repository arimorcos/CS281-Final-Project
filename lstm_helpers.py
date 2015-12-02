import numpy as np


def evaluate_network(network, data_butler, test_batch=500):
    """
    Evaluates the network performance on the training set.
    :param network: configured lstm_rnn network
    :param data_butler: data_butler object
    :param test_batch: the batch size for the tests
    :return:
        network_output: network_output_size x num_test examples array containing network output for each example
        answers: num_test_examples vector containing index of each correct exmaple
    """

    # Reinitialize data_butler
    data_butler.pull_from_test()
    data_butler.reset_test_schedule()

    # Get number of test examples
    num_test_queries = data_butler.test_queries

    # Set batch size
    data_butler.set_batch_size(test_batch)

    # Get number of iterations
    num_batches = np.ceil(float(num_test_queries)/test_batch)

    # Initialize
    network_output = np.array([])
    answers = np.array([])

    # Loop through each iteration
    for batch in range(num_batches):

        # Get data
        batch_vectors, batch_lengths, batch_answers = data_butler.offer_data()

        # Process
        batch_output = network.process(batch_vectors, batch_lengths, batch_answers)

        # Get answers as vector
        batch_answers = batch_answers.argmax(axis=0)

        # Concatenate
        network_output = np.hstack([network_output, batch_output])
        answers = np.vstack((answers, batch_answers))

        # Advance schedule
        data_butler.advance_schedule()

    # Reset to train mode
    data_butler.pull_from_train()

    # Return
    return network_output, answers