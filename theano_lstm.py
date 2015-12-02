import theano
import theano.tensor as T
from lstm_network_components import LSTM_stack, soft_reader
from network_optimizers import adadelta_fears_committment, adam_loves_theano
import sys
import numpy as np
import cPickle
import os
import re
import warnings


class lstm_rnn:
    """The full input to output network"""

    def __init__(self, inp_dim, layer_spec_list, final_output_size,
                 dropout=0.2, log_dir=None, init_train=None, save_weights_every=1,
                 b_i_offset=0., b_f_offset=0., b_c_offset=0., b_o_offset=0., b_y_offset=0.,
                 scale_down=0.9):
        """
        :param inp_dim: dimensionality of network input as a scalar
        :param layer_spec_list: List of 2-element tuples. Each tuple represents a layer in the network. The elements of
            tuples correspond to (num_hidden, num_output)
        :param final_output_size: scalar specifying the dimensionality of the total network output
        :param dropout: Fraction of neurons to dropout for each training iteration
        :param log_dir: Optional parameter to specify the log directory. Log directory must be set before proceeding though
        :param init_train: Optional paramater to specify a training function to initialize (supported: 'adam', 'adadelta')
        :param save_weights_every: Optional parameter to specify how often (in training steps) to save the network weights
        """
        # Initialize some parameters
        self.curr_params = None
        self.__adam_initialized = False
        self.__adadelta_initialized = False

        # Get you your LSTM stack
        self.LSTM_stack = LSTM_stack(inp_dim, layer_spec_list, dropout=dropout)
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

        # Get you your softmax readout
        self.soft_reader = soft_reader(LSTM_out_size, final_output_size)

        # Initialize weights
        self.initialize_network_weights(b_i_offset=b_i_offset, b_f_offset=b_f_offset, b_c_offset=b_c_offset,
                                        b_o_offset=b_o_offset, b_y_offset=b_y_offset, scale_down=scale_down)

        # Create the network graph
        self.create_network_graph()
        
        # Initialize training functions if requested
        self.__adam_initialized = False
        self.__adadelta_initialized = False
        if init_train == 'adam':
            self.initialize_training_adam()
        elif init_train == 'adadelta':
            self.initialize_training_adadelta()
        elif init_train is not None:
            warnings.warn('WARNING! Unable to initialize training function. Manually call a .initialize_training_*() ' \
                   'function before training.')

        self.steps_since_last_save = 0
        self.save_weights_every = save_weights_every

    def create_network_graph(self):

        # Input is a 3D stack of sequence represented by a matrix, treated as size = (max_seq_len, n_dim, n_examples)
        input_sequence = T.tensor3(name='inp', dtype=theano.config.floatX)
        
        # To fit in a matrix, sequences are zero-padded. So, we need the sequence lengths for each example.
        seq_lengths = T.ivector('seq_lengths')

        # Target is a onehot encoding of the correct answers for each example, treated as size = (n_options, n_examples)
        targets = T.matrix('targets', dtype=theano.config.floatX)

        # Through the LSTM stack, then soft max
        y, i, f, c, o, h = self.LSTM_stack.process(input_sequence, seq_lengths)
        p = self.soft_reader.process(y)

        # Give this class a process function
        self.process_unmodified_weights = theano.function([input_sequence, seq_lengths], p)

        # This gives access to the hidden activations
        self.hidden_activations = theano.function([input_sequence], [i, f, c, o, h])

        # Cost is based on the probability given to each entity
        cost = T.mean(T.nnet.binary_crossentropy(p, targets))

        ### For creating easy functions ###
        self.__p = p
        self.__cost = cost
        self.__inp_list = [input_sequence, seq_lengths, targets]
        self.__param_list = self.LSTM_stack.list_params() + self.soft_reader.list_params()
        self.__mask_list = self.LSTM_stack.list_masks() + self.soft_reader.list_masks()
        # self.__param_list = self.LSTM_stack.list_params()
        # self.__mask_list = self.LSTM_stack.list_masks()

        # For just getting your cost on a training example
        self.cost = theano.function(self.__inp_list, self.__cost)

    def process(self, input_sequence, seq_lengths):

        old_W_y = []
        for ind, layer in enumerate(self.LSTM_stack.layers):

            old_W_y.append(layer.W_y.get_value())
            layer.W_y.set_value(old_W_y[ind]*(1-self.dropout))

        output = self.process_unmodified_weights(input_sequence, seq_lengths)

        for ind, layer in enumerate(self.LSTM_stack.layers):
            layer.W_y.set_value(old_W_y[ind])

        return output

    def generate_masks(self):
        self.LSTM_stack.generate_masks()

    def initialize_training_adam(self, grad_max_norm=5, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.adam_helpers, self.adam_train, self.adam_param_list, self.adam_hyperparam_list, self.adam_grads =\
            adam_loves_theano(self.__inp_list, self.__cost, self.__param_list, self.__mask_list,
                              grad_max_norm=grad_max_norm, alpha=alpha, beta1=beta1, beta2=beta2, epsilon=epsilon)
        self.__adam_initialized = True

    def initialize_training_adadelta(self, rho=0.95, epsilon=1e-6):
        self.adadelta_helpers, self.adadelta_train, self.adadelta_param_list =\
            adadelta_fears_committment(self.__inp_list, self.__cost, self.__param_list, self.__mask_list,
                                       rho=rho, epsilon=epsilon)
        self.__adadelta_initialized = True

    def reset_adam(self, grad_max_norm=5, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        if not self.__adam_initialized:
            raise BaseException("Adam is not currently initialized")

        for adam_param in self.adam_param_list:
            for obj in adam_param:
                obj.set_value(
                        np.zeros(obj.get_value().shape)
                        .astype(theano.config.floatX))

        # Update hyperparameters
        self.adam_hyperparam_list.set_value(
            np.array([grad_max_norm, alpha, beta1, beta2, epsilon])
            .astype(theano.config.floatX)
        )


    def reset_adadelta(self):
        if not self.__adadelta_initialized:
            raise BaseException("Adadelta is not currently initialized")

        for adam_param in self.adam_param_list:
            for obj in adam_param:
                obj.set_value(
                    np.zeros(obj.get_value().shape)
                    .astype(theano.config.floatX))

    def set_dropout(self, dropout):
        answer = raw_input("Changing dropout will reset network. Continue? y/n")
        if answer == 'y':
            self.dropout = dropout
            for layer in self.LSTM_stack.layers:
                layer.dropout = dropout
            self.generate_masks()
        elif answer == 'n':
            return
        else:
            raise IOError("Input not understood")

    def initialize_network_weights(self, b_i_offset=0., b_f_offset=0., b_c_offset=0., b_o_offset=0., b_y_offset=0., scale_down=0.9):
        """
        initializes all network weights and re-initializes training functions if previously initialized
        """

        # Initialize stack and softreader weights
        self.LSTM_stack.initialize_stack_weights(b_i_offset, b_f_offset, b_c_offset, b_o_offset, b_y_offset)
        self.soft_reader.initialize_weights()

        # Reinitialize training functions
        if self.__adam_initialized:
            self.reset_adam()
        if self.__adadelta_initialized:
            self.reset_adadelta()
            
        # Scale down
        for p in self.list_all_params():
            p.set_value(p.get_value() * scale_down)

        # Normalize soft reader
        self.do_max_norm_reg()
        # self.soft_reader_norm()

        self.curr_epoch = 0
        self.curr_params = [p.get_value() for p in self.list_all_params()]

    def list_all_params(self):
        return self.LSTM_stack.list_params() + self.soft_reader.list_params()

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

        # Save new specs
        self.save_model_specs()

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
            all_params = self.list_all_params()
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
            all_params = self.list_all_params()

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

    def save_model_specs(self, save_file="model_specs"):
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
        save_file_pkl = os.path.join(self.log_dir, save_file + '.pkl')

        # Constructe dictionary of model parameters
        layer_params = []
        for layer in self.LSTM_stack.layers:
            layer_params.append({'num_hidden': layer.num_hidden,
                                 'num_outputs': layer.num_outputs,
                                 'num_inputs': layer.num_inputs})

        save_dict = {'final_output_size': self.final_output_size,
                     'layer_params': layer_params,
                     'dropout': self.dropout,
                     'num_layers': len(self.LSTM_stack.layers)}

        # save numpy file
        with open(save_file_pkl, mode='wb') as f:
            cPickle.dump(save_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)

        # save txt file
        save_file_txt = os.path.join(self.log_dir, save_file + '.txt')
        with open(save_file_txt, mode='wb') as f:
            f.write("Log directory: {} \n".format(self.get_log_dir()))
            f.write("Final output size: {} \n".format(self.final_output_size))
            f.write("Dropout rate: {} \n".format(self.dropout))
            f.write("Number of layers: {} \n".format(len(self.LSTM_stack.layers)))
            for ind, layer in enumerate(self.LSTM_stack.layers):
                f.write("Layer {}: \n".format(ind + 1))
                f.write("\tNumber of input units: {} \n".format(layer.num_inputs))
                f.write("\tNumber of hidden units: {} \n".format(layer.num_hidden))
                f.write("\tNumber of output units: {} \n".format(layer.num_outputs))
    
    def get_save_freq(self):
        return self.save_weights_every
    
    def set_save_freq(self,save_freq):
        save_freq = int(save_freq)
        if save_freq < 1:
            save_freq = 1
        self.save_weights_every = save_freq

    def get_param_diff(self):
        """
        Gets the updated parameter list, compares to the currnent list, and sets the new one as curr_weights
        Returns
        -------
        List of parameter update sizes
        """

        # Get new parameters
        old_params = self.curr_params
        new_params = [p.get_value() for p in self.list_all_params()]

        # Initialize difference
        param_diff = []

        # Get difference
        for old, new in zip(old_params, new_params):
            temp_diff = new - old
            param_diff.append(temp_diff)

        # set new parameters
        self.curr_params = new_params

        return param_diff

    @staticmethod
    def project_norm(in_vec, max_norm=3.5, do_always=False):
        """
        Helper function to project a vector to a sphere with a fixed norm while maintaining the direction. Will only
        apply projection if the current vector norm is greater than max_norm
        :param in_vec: the input vector
        :param max_norm: the maximum norm. Any vectors with norm > max_norm will be projected to max_norm
        :return: Normalized vector
        """
        curr_norm = np.linalg.norm(in_vec)
        if curr_norm > max_norm or do_always:
            scale_fac = max_norm/curr_norm
            in_vec *= scale_fac
        return in_vec

    def do_max_norm_reg(self, max_norm=3.5):
        """
        Applies max-norm regularization to each of the rows of the weight matrices (the inputs to each hidden unit)
        :param max_norm: the maximum norm. Any vectors with norm > max_norm will be projected to max_norm.
        :return: None
        """
        # Get list of all parameters
        all_params = self.LSTM_stack.list_params() + self.soft_reader.list_params()

        # loop through and perform regularization
        for param in all_params:

            # Check if parameter is vector, if so skip max-norm regularization
            if np.any(param.broadcastable):  # if a vector, some dimensions will be broadcastable
                continue

            # Get parameter value
            old_value = param.get_value()
            new_value = np.empty(shape=old_value.shape)

            # # loop through each column
            # for col in range(old_value.shape[1]):
            #     new_value[:, col] = self.project_norm(old_value[:, col], max_norm=max_norm)

            # loop through each row
            for row in range(old_value.shape[0]):
                new_value[row, :] = self.project_norm(old_value[row, :], max_norm=max_norm)

            # set value
            param.set_value(new_value.astype(theano.config.floatX))

    def soft_reader_norm(self):
        w = self.soft_reader.w
        old_value = w.get_value()
        num_rows = old_value.shape[0]
        for row in range(num_rows):
            old_value[row, :] = self.project_norm(old_value[row, :], max_norm=1, do_always=True)

        w.set_value(old_value)

    def adadelta_step(self, sequence, seq_length, target):
        """
        Calls the step function using adadelta and writes parameters
        :param sequence: input sequence
        :param seq_length: sequence length
        :param target: target variable
        :return:
            cost: The evaluated cost function
            param_diff: List of each parameter containing arrays of the change for this step
        """

        # Generate new dropout mask
        self.generate_masks()

        # Step and train
        self.adadelta_helpers(sequence, seq_length, target)
        cost = self.adadelta_train(sequence, seq_length, target)

        # perform max norm regularization
        self.do_max_norm_reg()
        # self.soft_reader_norm()

        # Get parameter difference
        param_diff = self.get_param_diff()

        # Write parameters if appropriate
        self.steps_since_last_save += 1
        if self.steps_since_last_save >= self.save_weights_every:
            self.write_parameters()
            self.steps_since_last_save = 0
        self.curr_epoch += 1

        return cost, param_diff

    def adam_step(self, sequence, seq_length, target):
        """
        Calls the step function using adam and writes parameters
        :param sequence: input sequence
        :param seq_length: sequence lengths
        :param target: target variable
        :return:
            cost: The evaluated cost function
            param_diff: List of each parameter containing arrays of the change for this step
        """

        # Generate new dropout mask
        self.generate_masks()

        # Step and train
        # cost = self.adam_step_train(sequence, seq_length, target)
        self.adam_helpers(sequence, seq_length, target)
        cost = self.adam_train(sequence, seq_length, target)

        # perform max norm regularization
        self.do_max_norm_reg()
        # self.soft_reader_norm()

        # Get parameter difference
        param_diff = self.get_param_diff()

        self.steps_since_last_save += 1
        if self.steps_since_last_save >= self.save_weights_every:
            self.write_parameters()
            self.steps_since_last_save = 0
        self.curr_epoch += 1
        return cost, param_diff
