import theano
import theano.tensor as T
import numpy as np


def adam_loves_theano(inp_list, cost, param_list, mask_list, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
    """
    adam: adaptive... momentum???

    Parameters
    ----------
    inp_list: List of Theano variables
        Whatever non-parameter things are needed to do a training step
    cost: Theano variable
        Objective fucntion to minimize
    param_list: List of Theano variables
        The variables that are changed for optimization
    mask_list: List of Theano variables
        The theano variable masks for each parameter
    [alpha]: {0.001}
        Training parameter: learning rate
    [beta1]: {0.9}
        Training parameter: decay rate for momentum
    [beta2]: {0.999}
        Training parameter: decay rate for velocity
    [epsilon]: {1e-7}
        Training parameter: i dunno.

    Outputs
    -------
    train_adam: A function that takes the inputs in inp_list and runs these two guys (which are created below),
        f_adam_helpers (updates helpers)
        f_adam_train (uses updated helpers to update parameters in param_list)

    """
    # Create 2 theano functions that will be called sequentially
    # The first one "updates" the shared variables that go into the calculation of the parameter update
    # The second one combines them into an update

    # Create the first function:
    # (These are going to be useful to precompute and store as a list):
    grads = [T.grad(cost, p) for p in param_list]
    # Initialize the helper variables, one for each parameter (this will only happen once and doesn't affect updates)
    Ts = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
          for p, g in zip(param_list, grads)]  # t term in adam
    Ms = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
          for p, g in zip(param_list, grads)]  # m term in adam
    Vs = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
          for p, g in zip(param_list, grads)]  # v term in adam

    # Define parameter list of shared variables
    adam_param_list = [Ts, Ms, Vs]

    # Define each of their update rules
    up_t = [(T_, T_+msk) for T_, msk in zip(Ts, mask_list)]
    up_m = [(M, msk*(beta1*M + (1-beta1)*g) + (1-msk)*M)
            for M, p, g, msk in zip(Ms, param_list, grads, mask_list)]
    up_v = [(V, msk*(beta2*V + (1-beta2)*(g**2)) + (1-msk)*V)
            for V, p, g, msk in zip(Vs, param_list, grads, mask_list)]

    # Combine this into a full update list
    up_h = up_t + up_m + up_v

    # Create that first function
    f_adam_helpers = theano.function(inp_list, cost, updates=up_h)

    # Create the second function (during training, this is called right after calling the first):
    # Compute, using the updated helper variables, the components of the parameter update equation
    # (updated by the call to f_adam_helpers, which will occurr during training)
    mHat = [m / (1-(beta1**t)) for m, t in zip(Ms, Ts)]
    vHat = [v / (1-(beta2**t)) for v, t in zip(Vs, Ts)]
    # Use them to update the parameters
    up_p = [(p, p - (alpha*mH / (T.sqrt(vH)+epsilon))*msk ) for p, mH, vH, msk in zip(param_list, mHat, vHat, mask_list)]
    # Create your training function with this update
    f_adam_train = theano.function(inp_list, cost, updates=up_p)
    
    # Combine these into a single function using this neat trick that Ari pointed out!
    # def train_adam( *args ):
    #     # Update helpers
    #     f_adam_helpers( *args )
    #     # Update parameters with updated helpers
    #     return f_adam_train( *args )

    return f_adam_helpers, f_adam_train, adam_param_list


def adadelta_fears_committment(inp_list, cost, param_list, mask_list, rho=.95, epsilon=1e-6):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    inp_list: List of Theano variables
        Whatever non-parameter things are needed to do a training step
    cost: Theano variable
        Objective fucntion to minimize
    param_list: List of Theano variables
        The variables that are changed for optimization
    [rho]: {0.95}
        Training parameter: decay rate
    [epsilon]: {1e-6}
        Training parameter: i dunno.

    Outputs
    -------
    train_adadelta: A function that takes the inputs in inp_list and runs these two guys (which are created below),
        f_adadelta_helpers (updates helpers)
        f_adadelta_train (uses updated helpers to update parameters in param_list)

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    ### = DESCRIPTION FROM LITERATURE

    # Initialize the helper variables, one for each parameter (this will only happen once and doesn't affect updates)
    grads = [T.grad(cost,p) for p in param_list]
    # Standard gradients: g_t
    zipped_grads = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
                    for p, g in zip(param_list, grads)]
    # Running expectation of squared update: E[ d[x]**2 ]_t
    running_up2 = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
                   for p, g in zip(param_list, grads)]
    # Running expectation of squared gradient: E[g**2]_t
    running_grads2 = [theano.shared(p.get_value()*np.zeros(1).astype(theano.config.floatX), broadcastable=g.broadcastable)
                      for p, g in zip(param_list, grads)]

    # Initialize parameter list
    adadelta_param_list = [zipped_grads, running_up2, running_grads2]


    ### Compute Gradient: g_t
    # Update rule for shared variables in zipped_grads (they just equal variables in grads)
    zgup = [(zg, T.grad(cost, p)) for zg, p in zip(zipped_grads, param_list)]

    ### Accumulate Gradient: E[g**2]_t = rho * E[g**2]_t-1  +  (1-rho) * (g_t)**2
    # Update rule for shared variables in running_grads2
    rg2up = [(rg2, (rho * rg2 + (1-rho) * (T.grad(cost, p) ** 2))*m + (1-m)*rg2)
             for rg2, m, p in zip(running_grads2, mask_list, param_list)]

    # Function that, when called, applies the two above update rules
    # (during training, this is called, then f_update is)
    f_adadelta_helpers = theano.function(inp_list, cost, updates=zgup+rg2up)


    ### Compute Update: d[x]_t = - [ RMS(d[x])_t-1 / RMS(g)_t ] * g_t
    # Create symbolic variable out of zipped_grads, running_up2, and running_grads2 for each parameter
    updir = [-T.sqrt(ru2 + epsilon) / T.sqrt(rg2 + epsilon) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]

    ### Accumulate Update: E[ d[x]**2 ]_t = rho * E[ d[x]**2 ]_t-1  +  (1-rho) * (d[x]_t)**2
    # Update rule for ru2up (whatever that is)
    ru2up = [(ru2, m*(rho * ru2 + (1-rho) * (ud ** 2)) + (1-m)*ru2)
             for ru2, m, ud in zip(running_up2, mask_list, updir)]

    ### Apply Update: x_t+1 = x_t + d[x]_t
    # Final update rule for parameter, combining all that
    # weight_updates = [m*ud for m, ud in zip(mask_list, updir)]
    param_up = [(p, p + m*ud) for p, m, ud in zip(param_list, mask_list, updir)]
    # param_up = [(p, p + wu) for p, wu in zip(param_list, weight_updates)]

    # Function to actually update the parameters (as well as ru2up)
    f_adadelta_train = theano.function(inp_list, cost, updates=ru2up + param_up)

    # Combine these into a single function using this neat trick that Ari pointed out!
    # def train_adadelta( *args ):
    #     # Update helpers
    #     f_adadelta_helpers( *args )
    #     # Update parameters with updated helpers
    #     return f_adadelta_train( *args )

    return f_adadelta_helpers, f_adadelta_train, adadelta_param_list


def i_hate_SGD(inp_list, cost,param_list, alpha=0.01):
    """
    SGD: but why???

    Parameters
    ----------
    inp_list: List of Theano variables
        Whatever non-parameter things are needed to do a training step
    cost: Theano variable
        Objective fucntion to minimize
    param_list: List of Theano variables
        The variables that are changed for optimization
    [alpha]: {0.001}
        Training parameter: learning rate

    Outputs
    -------
    train_SGD: function
        Uses updated helpers to update parameters in param_list

    """
    # This is so straightforward I should punch you if you don't understand.
    update_rules = [(p, p-T.grad(cost, p)*alpha) for p in param_list]
    train_SGD = theano.function(inp_list, cost, updates=update_rules)
    # Did you get it? Because if not you deserve punches.
    return train_SGD


# def rmsprop(lr, tparams, grads, x, mask, y, cost):
#     """
#     A variant of  SGD that scales the step size by running average of the
#     recent step norms.

#     Parameters
#     ----------
#     lr : Theano SharedVariable
#         Initial learning rate
#     tpramas: Theano SharedVariable
#         Model parameters
#     grads: Theano variable
#         Gradients of cost w.r.t to parameres
#     x: Theano variable
#         Model inputs
#     mask: Theano variable
#         Sequence mask
#     y: Theano variable
#         Targets
#     cost: Theano variable
#         Objective fucntion to minimize

#     Notes
#     -----
#     For more information, see [Hint2014]_.

#     .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
#        lecture 6a,
#        http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
#     """

#     zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
#                                   name='%s_grad' % k)
#                     for k, p in tparams.iteritems()]
#     running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
#                                    name='%s_rgrad' % k)
#                      for k, p in tparams.iteritems()]
#     running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
#                                     name='%s_rgrad2' % k)
#                       for k, p in tparams.iteritems()]

#     zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
#     rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
#     rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
#              for rg2, g in zip(running_grads2, grads)]

#     f_grad_shared = theano.function([x, mask, y], cost,
#                                     updates=zgup + rgup + rg2up,
#                                     name='rmsprop_f_grad_shared')

#     updir = [theano.shared(p.get_value() * numpy_floatX(0.),
#                            name='%s_updir' % k)
#              for k, p in tparams.iteritems()]
#     updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
#                  for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
#                                             running_grads2)]
#     param_up = [(p, p + udn[1])
#                 for p, udn in zip(tparams.values(), updir_new)]
#     f_update = theano.function([lr], [], updates=updir_new + param_up,
#                                on_unused_input='ignore',
#                                name='rmsprop_f_update')

#     return f_grad_shared, f_update