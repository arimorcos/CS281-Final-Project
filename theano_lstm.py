import theano
import theano.tensor as T
from lstm_network_components import LSTM_stack, soft_reader
from network_optimizers import adadelta_fears_committment, adam_loves_theano
import sys
import cPickle
import os
import re
import warnings


class lstm_rnn:
    """The full input to output network"""

    def __init__(self, inp_dim, layer_spec_list, final_output_size,
                 dropout=0.8, log_dir=None):
        """
        :param inp_dim: dimensionality of network input as a scalar
        :param layer_spec_list: List of 2-element tuples. Each tuple represents a layer in the network. The elements of
            tuples correspond to (num_hidden, num_output)
        :param final_output_size: scalar specifying the dimensionality of the total network output
        :param dropout: Fraction of neurons to use for each training iteration
        :param log_dir: Optional parameter to specify the log directory. Log directory must be set before proceeding
            though.
        """
        # Get you your LSTM stack
        self.LSTM_stack = LSTM_stack(inp_dim, layer_spec_list)
        LSTM_out_size = 0
        for L in layer_spec_list:
            LSTM_out_size += L[1]

        # store parameters
        self.dropout = dropout
        self.final_output_size = final_output_size

        # set log_dir if provided
        if log_dir is not None:
            try:
                self.set_log_dir(log_dir)
            except TypeError as e:
                warnings.warn("Cannot interpret log_dir: {}. \nRaised following exception: {}".
                              format(log_dir, e))
                self.log_dir = None
        else:
            self.log_dir = None

        # Initialize weights
        self.LSTM_stack.initialize_stack_weights()

        # Get you your softmax readout
        self.soft_reader = soft_reader(LSTM_out_size, final_output_size)

        # Create the network graph
        self.create_network_graph()

        # initialize the training functions
        self.initialize_training_functions()

        self.curr_epoch = 0

    def create_network_graph(self):

        # Input is a 3D stack of sequence represented by a matrix, treated as size = (max_seq_len, n_dim, n_examples)
        input_sequence = T.tensor3('inp')
        
        # To fit in a matrix, sequences are zero-padded. So, we need the sequence lengths for each example.
        seq_lengths = T.ivector('seq_lengths')

        # Target is a onehot encoding of the correct answers for each example, treated as size = (n_options, n_examples)
        if theano.config.device == 'gpu':
            targets = T.fmatrix('targets')
        else:
            targets = T.dmatrix('targets')


        # Through the LSTM stack, then soft max
        y = self.LSTM_stack.process(input_sequence, seq_lengths)
        p = self.soft_reader.process(y)

        # Give this class a process function
        self.process = theano.function([input_sequence, seq_lengths], p)

        # Cost is based on the probability given to each entity
        cost = T.sum( T.nnet.binary_crossentropy( p, targets ) )

        ### For creating easy functions ###
        self.__p = p
        self.__cost = cost
        self.__inp_list = [input_sequence, seq_lengths, targets]
        self.__param_list = self.LSTM_stack.list_params() + self.soft_reader.list_params()

        # For just getting your cost on a training example
        self.cost = theano.function(self.__inp_list, self.__cost)

    def initialize_training_functions(self):
        # For making training functions
        #adam
        self.adam_step_train =\
            adam_loves_theano(self.__inp_list, self.__cost, self.__param_list)

        #adadelta
        self.adadelta_step_train =\
            adadelta_fears_committment(self.__inp_list, self.__cost, self.__param_list)

    def initialize_network_weights(self):
        self.LSTM_stack.initialize_stack_weights()

    def set_log_dir(self, log_dir):
        """
        Sets the current log directory
        :param log_dir: path to log directory
        :return: None
        """
        self.log_dir = log_dir

        if os.path.isdir(log_dir):
            # Get file list
            file_list = os.listdir(log_dir)

            # Subset to epoch files
            epoch_string = 'Epoch_\d{4}_weights'
            epoch_files = [item for item in file_list if re.search(epoch_string, item)]

            # Get most up to date epoch and ask if files are present
            if epoch_files:

                # Get list of epoch strings
                epoch_nums = []
                for x in epoch_files:
                    match = re.search('((?<=Epoch\_)(\d{4})(?=\_weights))', x)
                    if match:
                        epoch_nums.append(int(match.group()))

                # get maximum epoch
                max_epoch = max(epoch_nums)

                # ask user
                answer = raw_input("Epoch files already exist. Maximum epoch is {}. Reset y/n?".format(max_epoch))
                if answer == 'y':
                    # Delete files
                    [os.remove(os.path.join(log_dir, x)) for x in epoch_files]
                elif answer == 'n':
                    # Set current epoch to max_epoch + 1
                    self.curr_epoch = max_epoch + 1
                else:
                    raise BaseException("Cannot parse answer.")

        else:
            os.mkdir(log_dir)

    def get_log_dir(self):
        """
        :return: Path to current log directory
        """
        if self.log_dir:
            return self.log_dir
        else:
            print "No log directory set."

    def write_parameters(self):
        """
        Writes the parameters to a new file in the current log folder
        :return: None
        """
        if self.log_dir is None:
            raise AttributeError("No log folder specified")

        # Create log file
        log_file = os.path.join(self.log_dir, "Epoch_{:04d}_weights.pkl".format(self.curr_epoch))

        with open(log_file, 'wb') as f:
            all_params = self.LSTM_stack.list_params() + self.soft_reader.list_params()
            for obj in all_params:
                cPickle.dump(obj.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)

    def set_parameters(self, param_file=None, epoch=None):
        """
        Loads parameters from the specified file and sets them
        :param param_file: file to load parameters from. If specified, loads a specific file.
        :param epoch: Epoch to load from. If specified, loads parameters from a given epoch in the current log
        folder. If both param_file and epoch provided, param_file will be used.
        :return: None
        """

        # Get load file
        if param_file:
            load_file = param_file
            if epoch:
                warnings.warn('Both param_file and epoch provided. Using param_file...')
            if not os.path.isfile(load_file):
                raise IOError('File {} not found. No parameters have been set'.format(load_file))
        elif epoch is not None:
            load_file = os.path.join(self.log_dir, 'Epoch_{:04d}_weights.pkl'.format(epoch))
            if not os.path.isfile(load_file):
                raise IOError('File corresponding to epoch {}: \"Epoch_{:04d}_weights.pkl\" not found.'.format(epoch,
                                                                                                           epoch))
        else:
            raise ValueError('Must provide param_file or epoch. No parameters have been set.')

        # load parameters
        with open(load_file, 'rb') as f:
            loaded_objects = []

            # Get the list of all objects
            all_params = self.LSTM_stack.list_params() + self.soft_reader.list_params()

            # loop through and load
            for obj in range(len(all_params)):
                loaded_objects.append(cPickle.load(f))

        # set parameters
        for ind, obj in enumerate(all_params):
            obj.set_value(loaded_objects[ind])

    def save_model(self, save_file):
        """
        Saves the entire network model
        :param save_file: path to file to save
        :return: None
        """

        # Get old recursion limit and set recursion limit to very high
        old_recursion_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(int(1e5))

        # save the model
        with open(save_file, mode='wb') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)

        # Reset old recursion limit
        sys.setrecursionlimit(old_recursion_limit)

    def save_model_specs(self, save_file="model_specs.pkl"):
        """
        Saves the specifications of the model: final output size, and for each layer: number of hidden units,
        layer input size, and layer output size
        :param save_file: path to file to save
        :return: None
        """

        # Check that the log directory is set
        if self.log_dir is None:
            raise AttributeError("No log folder specified")

        # Join path to log directory
        save_file = os.path.join(self.log_dir, save_file)

        # Constructe dictionary of model parameters
        layer_params = []
        for layer in self.LSTM_stack.layers:
            layer_params.append({'num_hidden': layer.num_hidden,
                                 'num_outputs': layer.num_outputs,
                                 'num_inputs': layer.num_inputs})

        save_dict = {'final_output_size': self.final_output_size,
                     'layer_params': layer_params}

        with open(save_file, mode='wb') as f:
            cPickle.dump(save_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)

    def adadelta_step(self, sequence, seq_length, target):
        """
        Calls the step function using adadelta and writes parameters
        :param sequence: input sequence
        :param seq_length: sequence length
        :param target: target variable
        :return: The evaluated cost function
        """
        cost = self.adadelta_step_train(sequence, seq_length, target)
        self.write_parameters()
        self.curr_epoch += 1
        return cost

    def adam_step(self, sequence, seq_length, target):
        """
        Calls the step function using adam and writes parameters
        :param sequence: input sequence
        :param seq_length: sequence lengths
        :param target: target variable
        :return: The evaluated cost function
        """
        cost = self.adam_step_train(sequence, seq_length, target)
        self.write_parameters()
        self.curr_epoch += 1
        return cost
