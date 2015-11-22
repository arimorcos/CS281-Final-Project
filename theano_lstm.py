class LSTM_layer:
    """A layer of an LSTM network"""
    def __init__(self,n_inp,n_hidden,n_out):
        self.n_inp = n_inp
        self.n_hidden = n_hidden
        self.n_out = n_out
        # LSTM layers have, for every hidden "unit" a unit and a corresponding memory cell
        # Memory cells include input, forget, and output gates as well as a value
        # There is also a set of outputs.
        # Fuck that's a lot of stuff.
        # (this should help):
        def init_w(n_in,n_out=n_hidden):
            return theano.shared( np.random.uniform(
                low = -1. / np.sqrt(n_in),
                high = 1. / np.sqrt(n_in),
                size = (n_out,n_in) ).astype(theano.config.floatX) )
        def init_b(n=n_hidden):
            return theano.shared( np.zeros(n).astype(theano.config.floatX) )
        # Initialize attributes for every weight of i
        self.w_i = init_w(n_inp+n_out + n_hidden + n_hidden) # (inp+prev_out + prev_hidden + prev_c)
        self.b_i = init_b()
        # Initialize attributes for every weight of f
        self.w_f = init_w(n_inp + n_hidden + n_hidden) # (inp + prev_hidden + prev_c)
        self.b_f = init_b()
        # Initialize attributes for every weight of c
        self.w_c = init_w(n_inp+n_out + n_hidden) # (inp+prev_out + prev_hidden)
        self.b_c = init_b()
        # Initialize attributes for every weight of o
        self.w_o = init_w(n_inp+n_out + n_hidden + n_hidden) # (inp+prev_out + prev_hidden + CURRENT_c)
        self.b_o = init_b()
        # Intialize attributes for weights of y (the real output)
        self.w_y = init_w(n_hidden,n_out)
        self.b_y = init_b(n_out)
        # Congrats. Now this is initialized.

    # Provide a list of all parameters to train
    def list_params(self):
        return [self.w_i,self.b_i,self.w_f,self.b_f,self.w_c,self.b_c,self.w_o,self.b_o,self.w_y,self.b_y]

    # Write methods for calculating the value of each of these playas at a given step
    def calc_i(self,combined_inputs):
        return T.nnet.sigmoid( T.dot( self.w_i, combined_inputs ) + self.b_i )
    def calc_f(self,combined_inputs):
        return T.nnet.sigmoid( T.dot( self.w_f, combined_inputs ) + self.b_f )
    def calc_c(self,prev_c,curr_f,curr_i,combined_inputs):
        return curr_f*prev_c + curr_i*T.tanh( T.dot( self.w_c, combined_inputs ) + self.b_c )
    def calc_o(self,combined_inputs):
        return T.nnet.sigmoid( T.dot( self.w_o, combined_inputs ) + self.b_o )
    def calc_h(self,curr_o,curr_c):
        return curr_o * T.tanh( curr_c )
    def calc_y(self,curr_h):
        return T.dot( self.w_y, curr_h ) + self.b_y
    # Put this together in a method for updating c, h, and y
    def step(self, inp, prev_c, prev_h, prev_y):
        i = self.calc_i( T.concatenate((inp,prev_y,prev_h,prev_c)) )
        f = self.calc_f( T.concatenate((inp,prev_h,prev_c)) )
        c = self.calc_c( prev_c, f, i, T.concatenate((inp,prev_y,prev_h)) )
        o = self.calc_o( T.concatenate((inp,prev_y,prev_h,c)) )
        h = self.calc_h( o, c )
        y = self.calc_y( h )
        return c, h, y