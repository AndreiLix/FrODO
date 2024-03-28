
import numpy as np
from matplotlib import pyplot as plt
import copy



def get_gradientVector_Autograd(f, x): 

    """
    Parameters
    -----------
    f: function
        An arbitrary function.
    x: array-like
        An array with the (multi-dimensional) point at which the subgradient vector of f is desired.
  
        
    Returns
    ------- 
    numpy array: shape (len(x), 1)
        Gradient vector of f at point x.
            Contains the partial derivatives of f with respect to x_1, ..., x_n.    

    """
    
    import jax
    import jax.numpy as jnp

    Df = np.zeros_like(x, dtype= np.float64)

    x1, x2 = x[0, 0], x[1, 0]

    f_partial_x1 = jax.grad(f, argnums= 0)
    f_partial_x2 = jax.grad(f, argnums= 1)

    Df[0, 0] = f_partial_x1(x1, x2)
    Df[1, 0] = f_partial_x2(x1, x2)
       
    return Df

def fractional_v2(x, len_memory, _lambda= 0.15):
    """
    _lambda: float in (0, 1)
        The fractional order to use. Must be in (0,1).

    This version of fractional weighting is the discreized version of the Riemann-Liouville version.
      Also found in "Discrete Fractional order PID controller". 
    """

    from scipy.special import gamma

    fract = ( 1 / gamma(_lambda) ) *  1 / ((len_memory-x) ** (1-_lambda))

    return fract


class DOptimizer():

    def __init__(self, models, fs_private, memory_len, beta_c, beta_g, beta_gm):
        
        self.models = models
        self.fs_private = fs_private
        self.len_memory = len(models)
        self.memory_len = memory_len
        self.beta_c = beta_c
        self.beta_g = beta_g
        self.beta_gm = beta_gm


    def step_withMemory(x, consensus_memory, gradient_memory, fs_private, scaled_memory= False, memory_profile= "exponential", b= 3, _lambda= 0.15, len_memory= 10, beta_c = 0.2, beta_cm= 0.04, beta_g= 0.6, beta_gm= 0.36):
        """

        Changes the states (x), consensus_memory and gradient_memory of our agents based on a step in the optimization process.

        Parameters
        ----------
        x: np array, shape (n_agents, n_params, 1).
            Agent states.

        consensus_memory: np array, shape (n_agents, n_params, len_memory)
            Memory of consensus error for each parameter of length len_memory.
            Memory encoded such that:
                consensus_memory[agent_i][param_i][0] = consensus error len_memory iterations ago.
                consensus_memory[agent_i][param_i][-1] = consensus error in the previous iteration.

        gradient_memory: np array, shape (n_agents, n_params, len_gradientMemory)
            Memory of past private subgradient values for each parameter of length len_gradientMemory.
                Memory encoded such that:
                gradient_memory[agent_i][param_i][0] = gradient len_memory iterations ago.
                gradient_memory[agent_i][param_i][-1] = gradient in the previous iteration.

        b, _lambda: 
            Scalar to parametrize the exponential and fractional memory profiles
                
            
        scaled_memory: bool
            If the memory feedback should be scaled by memory memory length.
        
            
        Returns
        --------
        x, consensus_memory, gradient_memory
            The updated angents's states (x), updated consensus_memory and gradient_memory
                                                                                                                                                                        
        """


        n_agents = len(fs_private)
        n_params = len(x[0])

        #---------------------------- first stage --------------------------------------

        # computing the terms used for the updates in this iteration

        consesus_term = np.zeros([n_agents, n_params,1])
        gradient_term = np.zeros([n_agents, n_params,1])


        for agent_i in range(n_agents):
            for param_i in range(n_params):
                consesus_term[agent_i][param_i][0] = beta_c * sum( [ (x[agent_j][param_i][0] - x[agent_i][param_i][0]) for agent_j in range(n_agents) if agent_j!= agent_i ] )
        
        for agent_i in range(n_agents):
            gradient_term[agent_i] = get_gradientVector_Autograd(fs_private[agent_i], x[agent_i])



        if memory_profile == "fractional_v2":
            memory_weights = np.array([fractional_v2(x, len_memory, _lambda) for x in range(len_memory)])
            # scaling values between 0 and 1
            memory_weights = memory_weights / max(memory_weights)



        z_g = np.zeros([n_agents, n_params, 1])


        for agent_i in range(n_agents):
            for param_i in range(n_params):           
                
                if scaled_memory:
                    z_g[agent_i][param_i][0] = (1/len_memory) * sum([ memory_weights[memory_i] * gradient_memory[agent_i][param_i][memory_i] for memory_i in range(len_memory)])

                if scaled_memory == False:
                    z_g[agent_i][param_i][0] = sum([ memory_weights[memory_i] * gradient_memory[agent_i][param_i][memory_i] for memory_i in range(len_memory)])



        # ================================================================


        #---------------------------- second stage --------------------------------------

        # updating x, consensus_memory and gradient_memory



        # NOTE: all terms in update must be shape (n_agents, n_params, 1)

        for agent_i in range(n_agents):
            for param_i in range(n_params):
                x[agent_i][param_i][0] =  x[agent_i][param_i][0] \
                                            + beta_c * consesus_term[agent_i][param_i][0] \
                                            - beta_g * gradient_term[agent_i][param_i][0] \
                                            - beta_gm * z_g[agent_i][param_i][0]
                
                # updating consensus mamory and gradient based on this iteration's sonsensus term and gradient descent term

                gradient_memory[agent_i][param_i][:-1] = gradient_memory[agent_i][param_i][1:]
                gradient_memory[agent_i][param_i][-1] = gradient_term[agent_i][param_i][0]


        return x, consensus_memory, gradient_memory
