import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np


class LSTM_layer:
    """A layer of an LSTM network"""

    def __init__(self, num_inputs=None, num_hidden=None, num_outputs=None, dropout=0.2):
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.dropout = dropout
        self.curr_mask = None
        self.null_mask = None
        self.null_output_mask = None
        self.initialized = False

    def set_weights(self, W_i, b_i, W_f, b_f, W_o, b_o, W_y, b_y):
        """
        :param W_i: LSTM input gate weights
        :param b_i: LSTM input gate bias
        :param W_f: LSTM forget gate weights
        :param b_f: LSTM forget gate bias
        :param W_o: LSTM output gate weights
        :param b_o: LSTM output gate bias
        :param W_y: Hidden layer output weights
        :param b_y: Hidden layer output bias
        :return: None
        """

        self.W_i = W_i
        self.b_i = b_i

        self.W_f = W_f
        self.b_f = b_f

        self.W_o = W_o
        self.b_o = b_o

        self.W_y = W_y
        self.b_y = b_y

    def initialize_weights(self, num_inputs=None, num_hidden=None, num_outputs=None):
        """
        :param num_inputs: number of input units
        :param num_hidden: number of hidden units
        :param num_outputs: number of output units
        :return: None
        """
        # Handle arguments. I dislike how Matlabian this is. But apparently default arguments are evaluated when a
        # function is declared, so you can't set the default argument to be a class attribute. Bummer.
        if num_inputs is None:
            num_inputs = self.num_inputs
        else:
            self.num_inputs = num_inputs

        if num_hidden is None:
            num_hidden = self.num_hidden
        else:
            self.num_hidden = num_hidden

        if num_outputs is None:
            num_outputs = self.num_outputs
        else:
            self.num_outputs = num_outputs

        # LSTM layers have, for every hidden "unit" a unit and a corresponding memory cell
        # Memory cells include input, forget, and output gates as well as a value
        # There is also a set of outputs.
        # Fuck that's a lot of stuff.
        # (this should help):

        # Initialize attributes for every weight of i
        W_i_size = (num_inputs + num_outputs + num_hidden + num_hidden,  # (inp + prev_out + prev_hidden + prev_c)
                    num_hidden)
        if self.initialized:
            self.reset_W(self.W_i)
            self.reset_b(self.b_i)
        else:
            self.W_i = self.__init_W__(*W_i_size)
            self.b_i = self.__init_b__(num_hidden)

        # Initialize attributes for every weight of f
        W_f_size = (num_inputs + num_hidden + num_hidden,  # (inp + prev_hidden + prev_c)
                    num_hidden)
        if self.initialized:
            self.reset_W(self.W_f)
            self.reset_b(self.b_f)
        else:
            self.W_f = self.__init_W__(*W_f_size)
            self.b_f = self.__init_b__(num_hidden)

        # Initialize attributes for every weight of c
        W_c_size = (num_inputs + num_outputs + num_hidden,  # (inp + prev_out + prev_hidden)
                    num_hidden)
        if self.initialized:
            self.reset_W(self.W_c)
            self.reset_b(self.b_c)
        else:
            self.W_c = self.__init_W__(*W_c_size)
            self.b_c = self.__init_b__(num_hidden)


        # Initialize attributes for every weight of o
        W_o_size = (num_inputs + num_outputs + num_hidden + num_hidden,  # (inp+prev_out + prev_hidden + CURRENT_c)
                    num_hidden)
        if self.initialized:
            self.reset_W(self.W_o)
            self.reset_b(self.b_o)
        else:
            self.W_o = self.__init_W__(*W_o_size)
            self.b_o = self.__init_b__(num_hidden)

        # Intialize attributes for weights of y (the real output)
        if self.initialized:
            self.reset_W(self.W_y)
            self.reset_b(self.b_y)
        else:
            self.W_y = self.__init_W__(num_hidden, num_outputs)
            self.b_y = self.__init_b__(num_outputs)

        # Initialize mask
        self.initialize_masks()

        self.initialized = True

        # Congrats. Now this is initialized.

    @staticmethod
    def __init_W__(n_in, n_out):
        # return theano.shared(np.random.uniform(
        #     low=-1. / np.sqrt(n_in),
        #     high=1. / np.sqrt(n_in),
        #     size=(n_out, n_in)).astype(theano.config.floatX))
        # return theano.shared(np.random.uniform(
        #     low=-1. / 4 * np.sqrt(6. / (n_in + n_out)),
        #     high=1. / 4 * np.sqrt(6. / (n_in + n_out)),
        #     size=(n_out, n_in)).astype(theano.config.floatX))
        return theano.shared(fan_in_out_uniform(n_in, n_out))
        # return theano.shared(ortho_weight(n_in, n_out))

    @staticmethod
    def __init_b__(n):
        # This is, effectively a vector, but we have to make it n-by-1 to enable broadcasting and batch processing
        return theano.shared( np.zeros((n,1)).astype(theano.config.floatX), broadcastable=(False,True) )

    @staticmethod
    def reset_W(w):
        w_shape = w.get_value().shape
        # w.set_value(np.random.uniform(
        #     low=-1. / 4 * np.sqrt(6. / (w_shape[1] + w_shape[0])),
        #     high=1. / 4 * np.sqrt(6. / (w_shape[1] + w_shape[0])),
        #     size=w_shape).astype(theano.config.floatX)
        # )
        w.set_value(fan_in_out_uniform(w_shape[1], w_shape[0]))
        # w.set_value(ortho_weight(w_shape[1], w_shape[0]))

    @staticmethod
    def reset_b(b):
        b_shape = b.get_value().shape
        b.set_value(np.zeros(b_shape).astype(theano.config.floatX))

    def initialize_masks(self):
        self.curr_mask = theano.shared(np.ones(shape=(1, self.num_hidden)).astype(theano.config.floatX),
                                       broadcastable=(True, False))
        # self.curr_mask = theano.shared(np.ones(shape=(self.num_hidden, 1)).astype(theano.config.floatX),
        #                                broadcastable=(False, True))
        self.null_mask = theano.shared(np.ones(shape=(self.num_hidden, 1)).astype(theano.config.floatX),
                                       broadcastable=(False, True))
        # self.null_mask = theano.shared(np.ones(shape=(1, self.num_hidden)).astype(theano.config.floatX),
        #                                broadcastable=(True, False))
        self.null_output_mask = theano.shared(np.ones(shape=(self.num_outputs, 1)).astype(theano.config.floatX),
                                       broadcastable=(False, True))

    def generate_masks(self):
        srng = RandomStreams()
        dropout_mask = np.random.binomial(n=1, p=(1 - self.dropout), size=(1, self.num_hidden)).astype(
            theano.config.floatX)
        # dropout_mask = np.random.binomial(n=1, p=(1 - self.dropout), size=(self.num_hidden, 1)).astype(
        #     theano.config.floatX)
        self.curr_mask.set_value(dropout_mask)

    def list_masks(self):
        return [self.null_mask, self.null_mask, self.null_mask, self.null_mask, self.null_mask, self.null_mask,
                self.null_mask, self.null_mask, self.curr_mask, self.null_output_mask]

    def list_params(self):
        # Provide a list of all parameters to train
        return [self.W_i, self.b_i, self.W_f, self.b_f, self.W_c, self.b_c, self.W_o, self.b_o, self.W_y, self.b_y]

    # Write methods for calculating the value of each of these playas at a given step
    def calc_i(self, combined_inputs):
        return T.nnet.sigmoid(T.dot(self.W_i, combined_inputs) + self.b_i)

    def calc_f(self, combined_inputs):
        return T.nnet.sigmoid(T.dot(self.W_f, combined_inputs) + self.b_f)

    def calc_c(self, prev_c, curr_f, curr_i, combined_inputs):
        return curr_f*prev_c + curr_i*T.tanh(T.dot(self.W_c, combined_inputs) + self.b_c)

    def calc_o(self, combined_inputs):
        return T.nnet.sigmoid(T.dot(self.W_o, combined_inputs) + self.b_o)

    def calc_h(self, curr_o, curr_c):
        return curr_o * T.tanh(curr_c)

    def calc_y(self, curr_h):
        return T.dot(self.W_y, T.transpose(self.curr_mask)*curr_h) + self.b_y
        # return T.dot(self.W_y, self.curr_mask*curr_h) + self.b_y
        # return T.dot(self.W_y, curr_h) + self.b_y

    def step(self, inp, prev_c, prev_h, prev_y):
        # Put this together in a method for updating c, h, and y
        i = self.calc_i(T.concatenate((inp, prev_y, prev_h, prev_c)))
        f = self.calc_f(T.concatenate((inp, prev_h, prev_c)))
        c = self.calc_c(prev_c, f, i, T.concatenate((inp, prev_y, prev_h)))
        o = self.calc_o(T.concatenate((inp, prev_y, prev_h, c)))
        h = self.calc_h(o, c)
        y = self.calc_y(h)

        return c, h, y, i, f, o


class LSTM_stack:
    """A stack of LSTMs"""

    def __init__(self, inp_dim, layer_spec_list, dropout=0.2):
        """
        Create each layer. Store them as a list.

        :param inp_dim: dimensionality of network input as a scalar
        :param layer_spec_list: List of 2-element tuples. Each tuple represents a layer in the network. The elements of
            tuples correspond to (num_hidden, num_output)
        :return: None
        """
        self.layers = []
        for K, spec in enumerate(layer_spec_list):
            # If the first layer, set the input dimensionality to the dimensionality of the input to the entire
            # stack. Otherwise, set it to the output of the previous layer.
            if K == 0:
                my_inps = inp_dim
            else:
                my_inps = layer_spec_list[K-1][1]

            self.layers = self.layers + [LSTM_layer(my_inps, spec[0], spec[1], dropout=dropout)]

    def initialize_stack_weights(self):
        """
        Initializes the weights for each layer in the stack
        :return: None
        """
        for layer in self.layers:
            layer.initialize_weights()

    def list_params(self):
        # Return all the parameters in this stack.... You sure?
        P = []
        for L in self.layers:
            P = P + L.list_params()

        return P

    def list_masks(self):
        # Return all the masks in this stack.... You sure? YES I"M SURE I'M AN ADULT!!!
        # I HATE YOU
        # YOU'RE NOT MY REAL DAD
        M = []
        for L in self.layers:
            M = M + L.list_masks()

        return M

    def generate_masks(self):
        for L in self.layers:
            L.generate_masks()

    def process(self, inp_sequences, seq_lengths):
        """
        This network component's symbolic graph. Full input -> output function performed by this component.
        This function takes/returns **Theano Variables**
        
        Parameters
        ------
        inp_sequences: tensor3() Variable
            Treated as size=(longest_sequence, input_dimension, n_examples)
        seq_lengths: ivector() Variable
            seq_lengths[i] = The shape[0] of inp_sequences[:,:,i] before zero-padding
            So, it is treated as size=(n_examples,)
            
        Returns
        -------
        Outputs at end of a given sequence, concatenated across layers
        
            
        """
        # Go through the whole input and return the concatenated outputs of the stack after it's all said and done
        outs = []
        all_I = []
        all_F = []
        all_C = []
        all_O = []
        all_H = []
        for K, layer in enumerate(self.layers):
            if K == 0:
                curr_seq = inp_sequences
            else:
                curr_seq = Y  # (from previous layer)
            
            # Initialize outputs C, H, and Y so that they support a variable number of examples
            n_ex = curr_seq.shape[2]
            out_init = [
                theano.tensor.alloc(np.zeros(1).astype(theano.config.floatX), layer.num_hidden,  n_ex),
                theano.tensor.alloc(np.zeros(1).astype(theano.config.floatX), layer.num_hidden,  n_ex),
                theano.tensor.alloc(np.zeros(1).astype(theano.config.floatX), layer.num_outputs, n_ex),
                None,
                None,
                None
                ]

            ([C,H,Y,I,F,O],updates) = theano.scan(fn=layer.step,
                                            sequences=curr_seq,
                                            outputs_info=out_init)
            
            # Return, for each example, only the final Y -- where "final" refers to the true sequence length
            outs = outs + [ Y[seq_lengths-1, :, T.arange(n_ex)] ]
            
            # Grow the hiddens
            all_I += [I]
            all_F += [F]
            all_C += [C]
            all_O += [O]
            all_H += [H]
            
        # Concatenate the grown hiddens
        I = T.concatenate( all_I, axis=1 )
        F = T.concatenate( all_F, axis=1 )
        C = T.concatenate( all_C, axis=1 )
        O = T.concatenate( all_O, axis=1 )
        H = T.concatenate( all_H, axis=1 )

        # Transpose so that we are consistent with things expecting n_dim-by-n_examples
        return T.transpose(T.concatenate( outs, axis=1 )), I, F, C, O, H

    
class soft_reader:
    """A softmax layer"""

    def __init__(self, num_inputs, num_outputs):
        # This is a simple layer, described just by a single weight matrix (no bias)
        # self.w = theano.shared(np.random.uniform(
        #         low=-1. / np.sqrt(num_inputs),
        #         high=1. / np.sqrt(num_inputs),
        #         size=(num_outputs, num_inputs) ).astype(theano.config.floatX))
        # self.w = theano.shared(np.random.uniform(
        #     low=-1. / 4 * np.sqrt(6. / (num_inputs + num_outputs)),
        #     high=1. / 4 * np.sqrt(6. / (num_inputs + num_outputs)),
        #     size=(num_outputs, num_inputs)).astype(theano.config.floatX))
        self.w = theano.shared(fan_in_out_uniform(num_inputs, num_outputs))
        # self.w = theano.shared(ortho_weight(num_inputs, num_outputs))

        # Create a null_mask
        self.null_mask = theano.shared(np.ones(shape=(num_outputs, 1)).astype(theano.config.floatX),
                                       broadcastable=(False, True))

    def initialize_weights(self):
        w_shape = self.w.get_value().shape
        # self.w.set_value(
        #     np.random.uniform(
        #         low=-1. / 4 * np.sqrt(6. / (w_shape[1] + w_shape[0])),
        #         high=1. / 4 * np.sqrt(6. / (w_shape[1] + w_shape[0])),
        #         size=w_shape
        #     ).astype(theano.config.floatX)
        # )
        self.w.set_value(fan_in_out_uniform(w_shape[1], w_shape[0]))
        # self.w.set_value(ortho_weight(w_shape[1], w_shape[0]))

    def list_masks(self):
        return [self.null_mask]

    def list_params(self):
        # Easy.
        return [self.w]

    def process(self, inp):
        """
        This network component's symbolic graph. Full input -> output function performed by this component.
        This function takes/returns **Theano Variables**
        
        Inputs
        ------
        inp_sequences: dmatrix() Variable
            Treated as size=(inp_dimension, num_examples)  <--- BATCH PROCESSING
            
        Outputs
        -------
        Outputs a Theano Variable
            Treated as size=(num_outputs, num_examples) <--- Each column sums to 1
        """
        # Do your soft max kinda thing.
        P = T.transpose( T.dot(self.w, inp) )
        return T.transpose( T.nnet.softmax(P) )


def ortho_weight(n_in, n_out):
    W = np.random.randn(n_out, n_in)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


def fan_in_out_uniform(n_in, n_out):
    return np.random.uniform(
        low=-1. / 4 * np.sqrt(6. / (n_in + n_out)),
        high=1. / 4 * np.sqrt(6. / (n_in + n_out)),
        size=(n_out, n_in)).astype(theano.config.floatX)
