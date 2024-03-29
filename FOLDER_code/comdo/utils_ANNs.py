import jax.numpy as jnp
from matplotlib import pyplot as plt
import copy




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

    def __init__(self, models, fs_private, len_memory= 100, _lambda= 0.15, beta_c = 0.2, beta_cm= 0.04, beta_g= 0.6, beta_gm= 0.36):
        
        """
        self.idx_layersWithWeights
            The 

        """

        self.models = models
        self.fs_private = fs_private
        self.n_agents = len(models)
        self.len_memory = len_memory
        self._lambda = _lambda
        self.beta_c = beta_c
        self.beta_g = beta_g
        self.beta_gm = beta_gm
        self.idx_layersWithWeights = []
        
        self.shape_gradient_memory = [self.n_agents, self.len_memory]
        self.shape_z_g = [self.n_agents]

        for i in range(len(models[0].layers)):
            if hasattr(models[0].layers[i], "weight"):
                self.idx_layersWithWeights.append(i)
                for dim in jnp.shape(models[0].layers[i]):
                    self.shape_gradient_memory.append(dim)
                    self.z_g.append(dim)



    def step_withMemory(self, models, grads_list):
        
        """

        Changes the states (x), consensus_memory and gradient_memory of our agents based on a step in the optimization process.

        Parameters
        ----------
        x: jnp array, shape (n_agents, n_params, 1).
            Agent states.

        gradient_memory: jnp array, shape (n_agents, self.len_memory, n_layers, <layer shape>)
            Memory of past private subgradient values for each parameter of length self.len_memory.
                Memory encoded such that:
                gradient_memory[agent_i][0] = gradients self.len_memory iterations ago.
                gradient_memory[agent_i][-1] = gradients in the previous iteration.

        _lambda: 
            Scalar to parametrize the fractional memory profile.
                
            
        scaled_memory: bool
            If the memory feedback should be scaled by memory memory length.
        
            
        Returns
        --------
        x, consensus_memory, gradient_memory
            The updated angents's states (x), updated consensus_memory and gradient_memory
                                                                                                                                                                        
        """


        n_agents = len(self.fs_private)
        n_params = len(x[0])

        #---------------------------- first stage --------------------------------------

        # computing the terms used for the updates in this iteration


        x = copy.deepcopy(models)

        consesus_term = jnp.zeros(self.shape_z_g)
        gradient_term = jnp.zeros(self.shape_z_g)

        
        for agent_i in range(n_agents):
            for layer_i in self.idx_layersWithWeights:

               gradient_term[agent_i][layer_i] = grads_list.layers[layer_i].weight


        memory_weights = jnp.array([fractional_v2(x, self.len_memory, self._lambda) for x in range(self.len_memory)])
        # scaling values between 0 and 1
        memory_weights = memory_weights / max(memory_weights)


        z_g = jnp.zeros([n_agents, n_params, 1])

        for agent_i in range(n_agents):

            aux_tensor = jnp.array([ memory_weights[memory_i] * self.gradient_memory[agent_i][memory_i] for memory_i in range(self.len_memory)])
            
            z_g[agent_i] = jnp.sum(aux_tensor, axis= 1)  # summing over the memory axis



        # ================================================================


        #---------------------------- second stage --------------------------------------

        # updating x, consensus_memory and gradient_memory



        # NOTE: all terms in update must be shape (n_agents, n_params, 1)

        for agent_i in range(n_agents):
            for layer_i in self.idx_layersWithWeights:

                x[agent_i][layer_i] =  x[agent_i][layer_i] \
                                            + self.beta_c * consesus_term[agent_i][layer_i] \
                                            - self.beta_g * gradient_term[agent_i][layer_i] \
                                            - self.beta_gm * z_g[agent_i][layer_i]
                
                # updating consensus mamory and gradient based on this iteration's sonsensus term and gradient descent term

                self.gradient_memory[agent_i][layer_i][:-1] = self.gradient_memory[agent_i][layer_i][1:]
                self.gradient_memory[agent_i][layer_i][-1] = gradient_term[agent_i][layer_i]


        return x
