import theano
import theano.tensor as T
from lstm_network_components import LSTM_stack, soft_reader
from lstm_optimizers import adadelta_fears_committment, adam_loves_theano
import csv


class lstm_rnn:
    """The full input to output network"""

    def __init__(self, inp_dim, layer_spec_list, final_output_size):
        """
        :param inp_dim: dimensionality of network input as a scalar
        :param layer_spec_list: List of 2-element tuples. Each tuple represents a layer in the network. The elements of
            tuples correspond to (num_hidden, num_output)
        """
        # Get you your LSTM stack
        self.LSTM_stack = LSTM_stack(inp_dim, layer_spec_list)
        LSTM_out_size = 0
        for L in layer_spec_list:
            LSTM_out_size += L[1]

        # Initialize weights
        self.LSTM_stack.initialize_stack_weights()

        # Get you your softmax readout
        self.soft_reader = soft_reader(LSTM_out_size, final_output_size)

        # Create the network graph
        self.create_network_graph()

        # initialize the training functions
        self.initialize_training_functions()

        self.log_file = None

    def create_network_graph(self):

        # Input is a sequence represented by a matrix
        input_sequence = T.dmatrix('inp')

        # Output is a scalar indicating the correct answer
        target = T.iscalar('target')

        # Through the LSTM stack, then soft max
        y = self.LSTM_stack.process(input_sequence)
        p = self.soft_reader.process(y)[0]

        # Give this class a process function
        self.process = theano.function([input_sequence], p)

        # Cost is based on the probability given to the correct answer
        # (this is like cross-entropy and still involves the whole w_v matrix because of softmax)
        cost = -T.log(p[target])

        ### For creating easy functions ###
        self.__p = p
        self.__cost = cost
        self.__inp_list = [input_sequence, target]
        self.__param_list = self.LSTM_stack.list_params() + self.soft_reader.list_params()

        # For just getting your cost on a training example
        self.cost = theano.function(self.__inp_list, self.__cost)

    def initialize_training_functions(self):
        # For making training functions
        #adam
        self.__f_adam_helpers, self.__f_adam_train =\
            adam_loves_theano(self.__inp_list, self.__cost, self.__param_list)

        #adadelta
        self.__f_adadelta_helpers, self.__f_adadelta_train =\
            adadelta_fears_committment(self.__inp_list, self.__cost, self.__param_list)

    def initialize_network_weights(self):
        self.LSTM_stack.initialize_stack_weights()

    def set_log_file(self, log_file):
        """
        Sets the current log file
        :param log_file: path to log file
        :return: None
        """
        self.log_file = log_file

    def get_log_file(self):
        """
        :return: Path to current log file
        """
        return self.log_file

    def write_parameters(self):
        if self.log_file is None:
            raise AttributeError("No log file created")

        with open(self.log_file, 'a+') as file_object:
            writer = csv.writer(file_object, delimiter=',')
            # writer.writerow()

    # These functions implements that sequential calling into one training step:
    def adam_step(self, S, T):
        self.__f_adam_helpers(S, T)
        return self.__f_adam_train(S, T)

    def adadelta_step(self, S, T):
        self.__f_adadelta_helpers(S, T)
        return self.__f_adadelta_train(S, T)