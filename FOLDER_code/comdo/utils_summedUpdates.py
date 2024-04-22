import numpy as np
from matplotlib import pyplot as plt
import copy


"""

utils_proper:
Same as utils, but with deepcopies instead of aliases for agent states. 
  Before this, I was uncounsciously doing centralized optimization.


NOTE:
- the states of every agent must be arrays containing only type float

TODO:
- replace np with jnp

"""


N_AGENTS = 4          # the number of agents

P = np.array(
    [ [[0.2, 0.1],
        [0.1, 0.2]],

      [[0.4, 0.1],
        [0.2, 0.4]],

      [[0.3, 0.1],
        [0.1, 0.2]],

      [[0.5, 0.1],
        [0.1, 0.2]],   
          ]
)

b = np.array(
    [ [[1],
        [8]],

      [[1],
        [1]],

      [[3],
        [1]],

      [[5],
        [1]],   
        ]
)

c = np.linspace(start= 0, stop= 1, num= N_AGENTS)      # c_i are chosen uniformly from [0, 1]




def f_global(x, fs_private):    # global objective function
    """    
    Returns
    -------
    output: float
        The output of our global objective function for the current state of the agents.
            The goal is to make this output as low as possible (see eq. 1).
    
    """


    private_outputs = [ f_i(x_i) for f_i, x_i in zip(fs_private, x) ]

    # print("Private Outputs:",  private_outputs)

    global_output = sum(private_outputs)

    return global_output




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




def step_sequential(x, z, fs_private, beta= 0.2, alpha= 3, a= 1):
    """
    Changes the states of our agents based on a step in the optimization process.

    Parameters
    ----------
    h: float > 0
        h used in the Euler method of get_subgradient()
    
    Returns
    --------
    x, z: tupple
        The updated angents's states (x) and updawted auxiliary states (z). 

    """

    n_agents = len(fs_private)


    #------------- first stage -----------------------
    
    # nodes exchanges states x_i and compute auxiliary states z_i.

    for i in range(n_agents):
        
        z[i] = z[i] + beta*sum( [ a*(x[i] - x[j]) for j in range(n_agents) if j!= i ]  )

    
    #------------- second stage --------------------  
    
    # nodes exchange auxiliary states z_i and update states x_i.
    
        
    for i in range(len(fs_private)):
        
        x[i] = x[i] + beta * sum( [ a*(x[j] - x[i]) for j in range(n_agents) if j!= i ] ) \
                    + beta * sum( [ a*(z[j] - z[i]) for j in range(n_agents) if j!= i ] ) \
                    - beta * alpha * get_gradientVector_Autograd(fs_private[i], x[i])
        
    return x, z




def step_parallel(x, z, fs_private, beta= 0.2, alpha= 3, a= 1):
    """
    Like step_sequential, but agent states get updated all at once, not sequentially.
        This is achieved by making an auxiliary variable (x_aux = list of length n_agents holding the states of all agents in the network)
            Disadvantages: higher memory requirement - on every step, there must be created a new variable with all the parameters in teh network.  

    Changes the states of our agents based on a step in the optimization process.

    Parameters
    ----------
    h: float > 0
        h used in the Euler method of get_subgradient()
    
    Returns
    --------
    x, z: tupple
        The updated angents's states (x) and updawted auxiliary states (z). 
                                                                                                                                                                    
    """

    n_agents = len(fs_private)


    gradient_vectors_Analytic = [ lambda x: np.array( 
                                            [ 
                                                2 * P[i][0][0]* x[0] + (P[i][1][0]+P[i][0][1])* x[1] + b[i][0], 
                                                2 * P[i][1][1]* x[1] + (P[i][1][0]+P[i][0][1])* x[0] + b[i][1]  
                                            ]

                                            )
                                for i in range(n_agents) ]


    #------------- first stage -----------------------
    
    # nodes exchanges states x_i and compute auxiliary states z_i.

    for i in range(n_agents):
        
        z[i] = z[i] + beta*sum( [ a*(x[i] - x[j]) for j in range(n_agents) if j!= i ]  )

    
    #------------- second stage --------------------  
    
    # nodes exchange auxiliary states z_i and update states x_i.
    

    x_aux = np.copy(x)      # using a copy of x in optimization, such that parameters get updated all at once, not sequentially
    
    for i in range(len(fs_private)):
        
        x[i] = x[i] + beta * sum( [ a*(x_aux[j] - x[i]) for j in range(n_agents) if j!= i ] ) \
                    + beta * sum( [ a*(z[j] - z[i]) for j in range(n_agents) if j!= i ] ) \
                    - beta * alpha * get_gradientVector_Autograd(fs_private[i], x[i])
                 
    return x, z




def constant(x): 
    con = 1
    return con


def linear(x): # slope redundant in linear, it can only have 1 slope, dependent on len(memory)
    lin = x
    return lin

def exponential(x, b= 3):
    assert b > 1, "Error: b must be > 1."
    exp = b ** x
    return exp

def fractional_v1(x, _lambda= 0.15): 
    """
    _lambda: float in (0, 1)
        The fractional order to use. Must be in (0,1).
    """

    from scipy.special import gamma

    binomial_coefficient = (gamma(_lambda+1)) / (gamma(x+1) * gamma(_lambda-x+1))
    fract = (-1) ** (x-1) * binomial_coefficient

    return fract

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


def step_withMemory(x, consensus_memory, gradient_memory, fs_private, scaled_memory= False, memory_profile= "exponential", b= 3, _lambda= 0.15, len_memory= 10, beta_c = 1, beta_cm= 0.04, beta_g= 0.6, beta_gm= 0.36):
  """
  Like step_sequential, but works with different memory profiles + more hyperparameter flexibility.

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

  # consesus_term = np.zeros([n_agents, n_params,1])
  gradient_term = np.zeros([n_agents, n_params,1])


  if memory_profile == "constant":
    memory_weights = np.array([constant(x) for x in range(1, len_memory + 1)])


  if memory_profile == "linear":
    memory_weights = np.array([linear(x) for x in range(1, len_memory + 1)])
    # scaling values between 0 and 1
    memory_weights = memory_weights / max(memory_weights)

  if memory_profile == "exponential":
    memory_weights = np.array([exponential(x, b) for x in range(1, len_memory + 1)])
    # scaling values between 0 and 1
    memory_weights = memory_weights / max(memory_weights)


  if memory_profile == "fractional_v1":
    memory_weights = np.array([fractional_v1(x, _lambda) for x in range(1, len_memory + 1)])
    # scaling values between 0 and 1
    memory_weights = memory_weights / max(memory_weights)
    #reversing array (only for fractional_v1)
    memory_weights = np.flip(memory_weights)

  if memory_profile == "fractional_v2":
    memory_weights = np.array([fractional_v2(x, len_memory, _lambda) for x in range(len_memory)])
    # scaling values between 0 and 1
    memory_weights = memory_weights / max(memory_weights)


  z_g = np.zeros([n_agents, n_params, 1])

  for agent_i in range(n_agents):

    gradient_term[agent_i] = get_gradientVector_Autograd(fs_private[agent_i], x[agent_i])
    for param_i in range(n_params):

      # consesus_term[agent_i][param_i][0] = (1/(n_agents-1)) * sum( [ (x[agent_j][param_i][0] - x[agent_i][param_i][0]) for agent_j in range(n_agents) if agent_j!= agent_i ] )
      # print(f"consensus term agent {agent_i}, param {param_i}: {consesus_term[agent_i][param_i][0]}")

      if scaled_memory:
        z_g[agent_i][param_i][0] = sum([ memory_weights[memory_i] * gradient_memory[agent_i][param_i][memory_i] for memory_i in range(len_memory)])

      if scaled_memory == False:
        z_g[agent_i][param_i][0] = sum([ memory_weights[memory_i] * gradient_memory[agent_i][param_i][memory_i] for memory_i in range(len_memory)])



  # ================================================================


  # #---------------------------- second stage --------------------------------------

  # # updating x, consensus_memory and gradient_memory



  # # NOTE: all terms in update must be shape (n_agents, n_params, 1)


  # for agent_i in range(n_agents):
  #   for param_i in range(n_params):

  #       # print(f"consensus term agent {agent_i}, param {param_i}: {consesus_term[agent_i][param_i][0]}")
  #       # print(f"gradient term agent {agent_i}, param {param_i}: {gradient_term[agent_i][param_i][0]}")

  #       x[agent_i][param_i][0] = x[agent_i][param_i][0] \
  #                                   + beta_c * consesus_term[agent_i][param_i][0] \
  #                                   - beta_g * gradient_term[agent_i][param_i][0] \
  #                                   - beta_gm * z_g[agent_i][param_i][0]
        
  #       # updating consensus mamory and gradient based on this iteration's sonsensus term and gradient descent term

  #       gradient_memory[agent_i][param_i][:-1] = gradient_memory[agent_i][param_i][1:]
  #       gradient_memory[agent_i][param_i][-1] = gradient_term[agent_i][param_i][0]



  # summing gradient term and memory feedback across agents
  summed_gradient_term = np.sum(gradient_term, axis= 0)
  summed_z_g = np.mean(z_g, axis= 0)

  # consensus step
  x_copy = np.copy(x)
  for agent_i in range(n_agents):
     x[agent_i] = np.copy(np.mean(x_copy, axis= 0)) + summed_gradient_term + summed_z_g


  # # descent step
  # for agent_i in range(n_agents):
  #   for param_i in range(n_params):
  #       x[agent_i][param_i][0] = x[agent_i][param_i][0] \
  #                                   - beta_g * gradient_term[agent_i][param_i][0] \
  #                                   - beta_gm * z_g[agent_i][param_i][0]



  gradient_memory[agent_i][param_i][:-1] = gradient_memory[agent_i][param_i][1:]
  gradient_memory[agent_i][param_i][-1] = gradient_term[agent_i][param_i][0]

  return x, consensus_memory, gradient_memory






def step_projected(x, consensus_memory, gradient_memory, fs_private, beta_c = 1, beta_g= 0.6, beta_pg= 0.36):
  """
  Optimization algorithm using projected gradients (forward and backward).
    
    Desired traits:
      - we get the perks of momentum (accelerated descent)
      - without the overshooting of momentum (when close to minimum, algorithm behaves like vanilla g.d.)

  Changes the states (x), consensus_memory and gradient_memory of our agents based on a step in the optimization process.

  Parameters
  ----------
  x: np array, shape (n_agents, n_params, 1).
    Agent states.

  consensus_memory: np array, shape (n_agents, n_params, len_memory)
    Memory of consensus error for each parameter of length len_memory.
      Memory encoded such that:
        consensus_memory[agent_i][param_i][0] = consensus error len_memory iterations ago.
        consensus_memory[agent_i][param_i][-1] = consensus error 1 iteration ago.

  gradient_memory: np array, shape (n_agents, n_params, len_gradientMemory)
      Memory of past private subgradient values for each parameter of length len_gradientMemory.
        Memory encoded such that:
          consensus_memory[agent_i][param_i][0] = consensus error len_memory iterations ago.
          consensus_memory[agent_i][param_i][-1] = consensus error 1 iteration ago.

  b, _lambda: 
    Scalar to parametrize the exponential and fractional memory profiles
          
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
      consesus_term[agent_i][param_i][0] = sum( [ (x[agent_j][param_i][0] - x[agent_i][param_i][0]) for agent_j in range(n_agents) if agent_j!= agent_i ] )
  
  for agent_i in range(n_agents):
      gradient_term[agent_i] = get_gradientVector_Autograd(fs_private[agent_i], x[agent_i])

 
  projG1 = np.zeros([n_agents, n_params,1])
  projG2 = np.zeros([n_agents, n_params,1])

  x_proj1 = np.copy(x)
  x_proj2 = np.copy(x)

  for agent_i in range(n_agents):
    for param_i in range(n_params):           
      
      x_proj1[agent_i][param_i] = x[agent_i][param_i] + (x[agent_i][param_i] - beta_g * gradient_memory[agent_i][param_i][-1])
      x_proj2[agent_i][param_i] = x[agent_i][param_i] - (x[agent_i][param_i] - beta_g * gradient_memory[agent_i][param_i][-1])


    projG1[agent_i] = get_gradientVector_Autograd(fs_private[agent_i], x_proj1[agent_i])
    projG2[agent_i] = get_gradientVector_Autograd(fs_private[agent_i], x_proj2[agent_i])
  

  # ================================================================


  #---------------------------- second stage --------------------------------------

  # updating x, consensus_memory and gradient_memory



  # NOTE: all terms in update must be shape (n_agents, n_params, 1)

  for agent_i in range(n_agents):
    for param_i in range(n_params):
        x[agent_i][param_i][0] =  x[agent_i][param_i][0] \
                                    + beta_c * consesus_term[agent_i][param_i][0] \
                                    - beta_g * gradient_term[agent_i][param_i][0] \
                                    - beta_pg * projG1[agent_i][param_i][0] \
                                    - beta_pg * projG2[agent_i][param_i][0]
        
        # updating consensus mamory and gradient based on this iteration's sonsensus term and gradient descent term
        # consensus_memory[agent_i][param_i][:-1] = consensus_memory[agent_i][param_i][1:]
        # consensus_memory[agent_i][param_i][-1] = consesus_term[agent_i][param_i][0]
    
        gradient_memory[agent_i][param_i][:-1] = gradient_memory[agent_i][param_i][1:]
        gradient_memory[agent_i][param_i][-1] = gradient_term[agent_i][param_i][0]


  return x, consensus_memory, gradient_memory




def optimize_IllDefinedHessian( stopping_condition : float = 0.002, max_iterations : int = 1000, scaled_memory= False, memory_profiles : list = ["exponential"], bs : list= [2], _lambdas: list = [2.5], lens_memory: list= [10], betas_c: list = [0.2], betas_cm: list = [0.04], betas_g: list = [0.6], betas_gm: list = [0.36], betas_pg: list = [0.36] ):
  """
  Returns
  -------
  dict

    dictionary of format { ( (x_1_initial, x_2_initial), memory_profile, optional_memoryProfileParameter, len_memory, beta_c, beta_cm, beta_g, beta_gm ) : n_iterationsUntilConvergence : int }
    
  """

  # --------------------- private objectives ------------------------

  def f1_ill(x1, x2):
      return 0.5 * ((2 - x1) ** 2) + 0.005 * (x2 ** 2)
  def f2_ill(x1, x2):
      return 0.5 * ((2 + x1) ** 2) + 0.005 * (x2 ** 2)
  def f3_ill(x1, x2):
      return 0.5 * (x1 ** 2) + 0.005 * ((2 - x2) ** 2)
  def f4_ill(x1, x2):
      return 0.5 * (x1 ** 2) + 0.005 * ((2 + x2) ** 2)
  
  # ==================================================================

  x_opt = np.array(
                [ [0] ,
                  [0] ]
                )

  # performance_dict_initialCondition = {}
  performance_dict = {}

  initial_conditions = ( (1., 0.), (0.86, 0.5), (0.5, 0.86), (0., 1.) )

  x_history = []

  for initial_condition in initial_conditions:



    last_iteration = 0


    for memory_profile in memory_profiles:

      if memory_profile == "projected":
        for beta_pg in betas_pg:
          for len_memory in lens_memory:
            for beta_c in betas_c:
              for beta_cm in betas_cm:
                for beta_g in betas_g:

                  # TODO: remove these conditions when Monte Carlo assessment done

                  
                  # if initial_condition == (1., 0.) and beta_g == 1:
                  #    continue
                  
                  # if initial_condition == (0., 1.) and beta_g == 0.1:
                  #    continue
                  
                  for beta_gm in betas_gm:

                    # assigning the same initial condition to all agents

                    fs_private = [f1_ill, f2_ill, f3_ill, f4_ill]
                    # x_history = []
                    x1 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x2 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x3 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x4 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x = [x1, x2, x3, x4]

                    # initializing integral terms. 
                    # NOTE: each term must be 0
                    # z1, z2, z3, z4 = 4 * [np.zeros_like(x1)]

                    # z = [z1, z2, z3, z4]

                    x_inLast2Iterations = [copy.deepcopy(x), copy.deepcopy(x)]
                    n_agents = len(fs_private)
                    n_params = len(x[0])
                    consensus_memory = np.zeros([n_agents, n_params, len_memory])
                    gradient_memory = np.zeros([n_agents, n_params, len_memory])

                    for iteration in range(max_iterations):
                          
                          x_history.append(copy.deepcopy(x))

                          last_iteration = last_iteration + 1
                          # np array of shape = (len(x_inLast2Iterations) * n_agents * n_params, )
                                #   contains the absolute difference of each parameter of the agents in the last 2 iterations from the optimum.
                          dif_fromOptimum = np.reshape( [ [ [abs(x_inLast2Iterations[k][i][j,0] - x_opt[j]) for j in range(len(x_inLast2Iterations[0][0]))] for i in range(len(x_inLast2Iterations[0])) ] for k in range(len(x_inLast2Iterations))], newshape= -1)

                          if all(dif < stopping_condition for dif in dif_fromOptimum):
                                
                                # x_history.append(copy.deepcopy(x))
                                # performance_dict[(initial_condition, memory_profile , b, _lambda, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                                performance_dict[(initial_condition, memory_profile, beta_c,  beta_g, beta_pg)] = iteration
                                break


                          # PROBLEM: uncommenting this line results in halving the number of iterations -> figure out why, maybe equivalent to doubling beta
                          # print(step_v3(x, z, fs_private, alpha= 3, beta= 0.2, subgradient= "autograd"))

                        #  x_history.append(copy.deepcopy(x))

                          x, consensus_memory, gradient_memory = step_projected(x, consensus_memory, gradient_memory, fs_private, beta_c = beta_c, beta_g= beta_g, beta_pg= beta_pg)
                          
                          x_inLast2Iterations[0] = copy.deepcopy(x_inLast2Iterations[1])
                          x_inLast2Iterations[1] = copy.deepcopy(x)

                          # print(x)


                    # performance_dict[(initial_condition, memory_profile, b, _lambda , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                    performance_dict[(initial_condition, memory_profile, beta_c,  beta_g, beta_pg)] = iteration
                    # print("performance_dict_initialCondition = ",  performance_dict_initialCondition)
                    
                    # performance_dict[( memory_profile , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = performance_dict_initialCondition



    for memory_profile in memory_profiles:

      if memory_profile == "exponential":
        for b in bs:
          for len_memory in lens_memory:
            for beta_c in betas_c:
              for beta_cm in betas_cm:
                for beta_g in betas_g:

                  # TODO: remove these conditions when Monte Carlo assessment done

                  
                  # if initial_condition == (1., 0.) and beta_g == 1:
                  #    continue
                  
                  # if initial_condition == (0., 1.) and beta_g == 0.1:
                  #    continue
                  
                  for beta_gm in betas_gm:

                    # assigning the same initial condition to all agents

                    fs_private = [f1_ill, f2_ill, f3_ill, f4_ill]
                    # x_history = []
                    x1 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x2 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x3 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x4 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x = [x1, x2, x3, x4]

                    # initializing integral terms. 
                    # NOTE: each term must be 0
                    # z1, z2, z3, z4 = 4 * [np.zeros_like(x1)]

                    # z = [z1, z2, z3, z4]

                    x_inLast2Iterations = [copy.deepcopy(x), copy.deepcopy(x)]
                    n_agents = len(fs_private)
                    n_params = len(x[0])
                    consensus_memory = np.zeros([n_agents, n_params, len_memory])
                    gradient_memory = np.zeros([n_agents, n_params, len_memory])

                    for iteration in range(max_iterations):

                          last_iteration = last_iteration + 1
                          # np array of shape = (len(x_inLast2Iterations) * n_agents * n_params, )
                                #   contains the absolute difference of each parameter of the agents in the last 2 iterations from the optimum.
                          dif_fromOptimum = np.reshape( [ [ [abs(x_inLast2Iterations[k][i][j,0] - x_opt[j]) for j in range(len(x_inLast2Iterations[0][0]))] for i in range(len(x_inLast2Iterations[0])) ] for k in range(len(x_inLast2Iterations))], newshape= -1)

                          if all(dif < stopping_condition for dif in dif_fromOptimum):
                                
                               # x_history.append(x)
                                # performance_dict[(initial_condition, memory_profile , b, _lambda, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                                performance_dict[(initial_condition, memory_profile, b, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = iteration


                                break


                          # PROBLEM: uncommenting this line results in halving the number of iterations -> figure out why, maybe equivalent to doubling beta
                          # print(step_v3(x, z, fs_private, alpha= 3, beta= 0.2, subgradient= "autograd"))

                         # x_history.append(copy.deepcopy(x))

                          

                          x, consensus_memory, gradient_memory = step_withMemory(x, consensus_memory, gradient_memory, fs_private, scaled_memory= scaled_memory, memory_profile= memory_profile, b= b, len_memory= len_memory, beta_c = beta_c, beta_cm= beta_cm, beta_g= beta_g, beta_gm= beta_gm)
                          
                          x_inLast2Iterations[0] = copy.deepcopy(x_inLast2Iterations[1])
                          x_inLast2Iterations[1] = copy.deepcopy(x)

                          # print(x)


                    # performance_dict[(initial_condition, memory_profile, b, _lambda , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                    performance_dict[(initial_condition, memory_profile, b, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = iteration
                    # print("performance_dict_initialCondition = ",  performance_dict_initialCondition)
                    
                    # performance_dict[( memory_profile , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = performance_dict_initialCondition


    for memory_profile in memory_profiles:

      if memory_profile in ("fractional_v1", "fractional_v2"):
        for _lambda in _lambdas:
          for len_memory in lens_memory:
            for beta_c in betas_c:
              for beta_cm in betas_cm:
                for beta_g in betas_g:

                  # TODO: remove these conditions when Monte Carlo assessment done

                  
                  # if initial_condition == (1., 0.) and beta_g == 1:
                  #    continue
                  
                  # if initial_condition == (0., 1.) and beta_g == 0.1:
                  #    continue
                  
                  for beta_gm in betas_gm:

                    # assigning the same initial condition to all agents

                    fs_private = [f1_ill, f2_ill, f3_ill, f4_ill]
                    # x_history = []
                    x1 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x2 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x3 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x4 = copy.deepcopy(np.array(
                          [ [initial_condition[0]] ,
                            [initial_condition[1]] ]
                          ))
                    x = [x1, x2, x3, x4]

                    x_inLast2Iterations = [copy.deepcopy(x), copy.deepcopy(x)]
                    n_agents = len(fs_private)
                    n_params = len(x[0])
                    consensus_memory = np.zeros([n_agents, n_params, len_memory])
                    gradient_memory = np.zeros([n_agents, n_params, len_memory])

                    for iteration in range(max_iterations):
                          
                          x_history.append(copy.deepcopy(x))
                          # print("x at iteration ", iteration, ":")
                          # print(x)

                          last_iteration = last_iteration + 1
                          # np array of shape = (len(x_inLast2Iterations) * n_agents * n_params, )
                                #   contains the absolute difference of each parameter of the agents in the last 2 iterations from the optimum.
                          dif_fromOptimum = np.reshape( [ [ [abs(x_inLast2Iterations[k][i][j,0] - x_opt[j]) for j in range(len(x_inLast2Iterations[0][0]))] for i in range(len(x_inLast2Iterations[0])) ] for k in range(len(x_inLast2Iterations))], newshape= -1)

                          if all(dif < stopping_condition for dif in dif_fromOptimum):
                                
                               # x_history.append(x)
                                # performance_dict[(initial_condition, memory_profile , b, _lambda, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                                performance_dict[(initial_condition, memory_profile, _lambda , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = iteration # , x_history

                                break

                          x, consensus_memory, gradient_memory = step_withMemory(x, consensus_memory, gradient_memory, fs_private, scaled_memory= scaled_memory, memory_profile= memory_profile, _lambda= _lambda, len_memory= len_memory, beta_c = beta_c, beta_cm= beta_cm, beta_g= beta_g, beta_gm= beta_gm)
                          
                          x_inLast2Iterations[0] = copy.deepcopy(x_inLast2Iterations[1])
                          x_inLast2Iterations[1] = copy.deepcopy(x)

                          # print(x)

                    performance_dict[(initial_condition, memory_profile, _lambda , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = iteration   # , x_history
                    

    for memory_profile in memory_profiles:

      if memory_profile in ("constant", "linear"):   # TODO: having 3 blocks to address this problem seems redundant, figure out a way to do it cleaner
        for len_memory in lens_memory:
          for beta_c in betas_c:
            for beta_cm in betas_cm:
              for beta_g in betas_g:

                # TODO: remove these conditions when Monte Carlo assessment done

                
                # if initial_condition == (1., 0.) and beta_g == 1:
                #     continue
                
                # if initial_condition == (0., 1.) and beta_g == 0.1:
                #     continue
                
                for beta_gm in betas_gm:

                  # assigning the same initial condition to all agents

                  fs_private = [f1_ill, f2_ill, f3_ill, f4_ill]
                  # x_history = []
                  x1 = copy.deepcopy(np.array(
                        [ [initial_condition[0]] ,
                          [initial_condition[1]] ]
                        ))
                  x2 = copy.deepcopy(np.array(
                        [ [initial_condition[0]] ,
                          [initial_condition[1]] ]
                        ))
                  x3 = copy.deepcopy(np.array(
                        [ [initial_condition[0]] ,
                          [initial_condition[1]] ]
                        ))
                  x4 = copy.deepcopy(np.array(
                        [ [initial_condition[0]] ,
                          [initial_condition[1]] ]
                        ))
                  x = [x1, x2, x3, x4]

                  x_inLast2Iterations = [copy.deepcopy(x), copy.deepcopy(x)]
                  n_agents = len(fs_private)
                  n_params = len(x[0])
                  consensus_memory = np.zeros([n_agents, n_params, len_memory])
                  gradient_memory = np.zeros([n_agents, n_params, len_memory])

                  for iteration in range(max_iterations):
                        
                        x_history.append(copy.deepcopy(x))
                        # print("x at iteration ", iteration, ":")
                        # print(x)


                        last_iteration = last_iteration + 1
                        # np array of shape = (len(x_inLast2Iterations) * n_agents * n_params, )
                              #   contains the absolute difference of each parameter of the agents in the last 2 iterations from the optimum.
                        dif_fromOptimum = np.reshape( [ [ [abs(x_inLast2Iterations[k][i][j,0] - x_opt[j]) for j in range(len(x_inLast2Iterations[0][0]))] for i in range(len(x_inLast2Iterations[0])) ] for k in range(len(x_inLast2Iterations))], newshape= -1)

                        if all(dif < stopping_condition for dif in dif_fromOptimum):
                              
                              # x_history.append(x)
                              # performance_dict[(initial_condition, memory_profile , b, _lambda, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                              performance_dict[(initial_condition, memory_profile, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = iteration  #, x_history

                              break

                        x, consensus_memory, gradient_memory = step_withMemory(copy.deepcopy(x), consensus_memory, gradient_memory, fs_private, scaled_memory= scaled_memory, memory_profile= memory_profile, len_memory= len_memory, beta_c = beta_c, beta_cm= beta_cm, beta_g= beta_g, beta_gm= beta_gm)
                        
                        x_inLast2Iterations[0] = copy.deepcopy(x_inLast2Iterations[1])
                        x_inLast2Iterations[1] = copy.deepcopy(x)

                        # print(x)

                  performance_dict[(initial_condition, memory_profile, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = iteration  #, x_history


  return performance_dict












def optimize_Rosenbrock( initial_condition= (np.random.uniform(0, 2), np.random.uniform(0, 2)), stopping_condition : float = 0.002, max_iterations : int = 1000, scaled_memory= False, memory_profiles : list = ["exponential"], bs : list= [2], _lambdas: list = [2.5], lens_memory: list= [10], betas_c: list = [0.2], betas_cm: list = [0.04], betas_g: list = [0.6], betas_gm: list = [0.36], betas_pg: list = [0.36] ):
  """

  Differences from the initial setup for testing sensitivity to initial conditions:
    - using the Rosenbrock function as the global objective  (a= 1, b= 100);
    - using only 2 agents instead of 4;
    - staring from the same initial condition
    - initial_condition = np.random.uniform(0, 2), np.random.uniform(0, 2)
    - global optimum (for f1 + f2) = (1., 1.)

    

  Parameters
  ----------
  initial_condition : tuple shape (2,)
    must contain type float
    default (np.random.uniform(0, 2), np.random.uniform(0, 2)) 
      sampling from inside the square (0, 2)x(0, 2)



  Returns
  -------
  dict

    dictionary of format { ( memory_profile, optional_memoryProfileParameter, len_memory, beta_c, beta_cm, beta_g, beta_gm ) : n_iterationsUntilConvergence : int }
    
  """

  # --------------------- private objectives ------------------------

    # their sum is the classic Rosenbrock function and its minimum is at (a, a**2).
  
  def f1(x1, x2, a=1.0, b=100.0):
    # Minima at x1=a
    return (a - x1)**2

  def f2(x1, x2, a=1.0, b=100.0):
      # Minima at x2=x1**2
      return (b * (x2 - x1**2)**2)
  
  # ==================================================================

  x_opt = np.array(
                [ [1.] ,
                  [1.] ]
                )

  # performance_dict_initialCondition = {}
  performance_dict = {}

  last_iteration = 0

  for memory_profile in memory_profiles:

    if memory_profile == "projected":
      for beta_pg in betas_pg:
        for len_memory in lens_memory:
          for beta_c in betas_c:
            for beta_cm in betas_cm:
              for beta_g in betas_g:

                for beta_gm in betas_gm:

                  # assigning the same initial condition to all agents

                  fs_private = [f1, f2]
                  x_history = []

                  x1 = copy.deepcopy(np.array(
                      [ [initial_condition[0]] ,
                        [initial_condition[1]] ]
                      ))
                                        
                  x2 = copy.deepcopy(np.array(
                      [ [initial_condition[0]] ,
                        [initial_condition[1]] ]
                      ))
                  
                  x = [x1, x2]

                  # initializing integral terms. 
                  # NOTE: each term must be 0
                  # z1, z2, z3, z4 = 4 * [np.zeros_like(x1)]

                  # z = [z1, z2, z3, z4]

                  x_inLast2Iterations = [copy.deepcopy(x), copy.deepcopy(x)]
                  n_agents = len(fs_private)
                  n_params = len(x[0])
                  consensus_memory = np.zeros([n_agents, n_params, len_memory])
                  gradient_memory = np.zeros([n_agents, n_params, len_memory])

                  for iteration in range(max_iterations):

                        last_iteration = last_iteration + 1
                        # np array of shape = (len(x_inLast2Iterations) * n_agents * n_params, )
                              #   contains the absolute difference of each parameter of the agents in the last 2 iterations from the optimum.
                        dif_fromOptimum = np.reshape( [ [ [abs(x_inLast2Iterations[k][i][j,0] - x_opt[j]) for j in range(len(x_inLast2Iterations[0][0]))] for i in range(len(x_inLast2Iterations[0])) ] for k in range(len(x_inLast2Iterations))], newshape= -1)

                        if all(dif < stopping_condition for dif in dif_fromOptimum):
                              
                              x_history.append(copy.deepcopy(x))
                              # performance_dict[(initial_condition, memory_profile , b, _lambda, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                              performance_dict[(initial_condition, memory_profile, beta_c,  beta_g, beta_pg)] = (iteration, x_history)
                              break


                        # PROBLEM: uncommenting this line results in halving the number of iterations -> figure out why, maybe equivalent to doubling beta
                        # print(step_v3(x, z, fs_private, alpha= 3, beta= 0.2, subgradient= "autograd"))

                        x_history.append(copy.deepcopy(x))

                        x, consensus_memory, gradient_memory = step_projected(x, consensus_memory, gradient_memory, fs_private, beta_c = beta_c, beta_g= beta_g, beta_pg= beta_pg)
                        
                        x_inLast2Iterations[0] = copy.deepcopy(x_inLast2Iterations[1])
                        x_inLast2Iterations[1] = copy.deepcopy(x)

                        # print(x)


                  # performance_dict[(initial_condition, memory_profile, b, _lambda , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                  performance_dict[(initial_condition, memory_profile, beta_c,  beta_g, beta_pg)] = (iteration, x_history)
                  # print("performance_dict_initialCondition = ",  performance_dict_initialCondition)
                  
                  # performance_dict[( memory_profile , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = performance_dict_initialCondition



  for memory_profile in memory_profiles:

    if memory_profile == "exponential":
      for b in bs:
        for len_memory in lens_memory:
          for beta_c in betas_c:
            for beta_cm in betas_cm:
              for beta_g in betas_g:
                for beta_gm in betas_gm:


                  fs_private = [f1, f2]
                  x_history = []

                  x1 = copy.deepcopy(np.array(
                      [ [initial_condition[0]] ,
                        [initial_condition[1]] ]
                      ))
                                        
                  x2 = copy.deepcopy(np.array(
                      [ [initial_condition[0]] ,
                        [initial_condition[1]] ]
                      ))
                  
                  x = [x1, x2]

                  x_inLast2Iterations = [copy.deepcopy(x), copy.deepcopy(x)]
                  n_agents = len(fs_private)
                  n_params = len(x[0])
                  consensus_memory = np.zeros([n_agents, n_params, len_memory])
                  gradient_memory = np.zeros([n_agents, n_params, len_memory])

                  for iteration in range(max_iterations):

                        last_iteration = last_iteration + 1
                        # np array of shape = (len(x_inLast2Iterations) * n_agents * n_params, )
                              #   contains the absolute difference of each parameter of the agents in the last 2 iterations from the optimum.
                        dif_fromOptimum = np.reshape( [ [ [abs(x_inLast2Iterations[k][i][j,0] - x_opt[j]) for j in range(len(x_inLast2Iterations[0][0]))] for i in range(len(x_inLast2Iterations[0])) ] for k in range(len(x_inLast2Iterations))], newshape= -1)

                        if all(dif < stopping_condition for dif in dif_fromOptimum):
                              
                              x_history.append(copy.deepcopy(x))
                              # performance_dict[(initial_condition, memory_profile , b, _lambda, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                              performance_dict[(initial_condition, memory_profile, b, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = (iteration, x_history)


                              break


                        # PROBLEM: uncommenting this line results in halving the number of iterations -> figure out why, maybe equivalent to doubling beta
                        # print(step_v3(x, z, fs_private, alpha= 3, beta= 0.2, subgradient= "autograd"))

                        x_history.append(copy.deepcopy(x))

                        x, consensus_memory, gradient_memory = step_withMemory(x, consensus_memory, gradient_memory, fs_private, scaled_memory= scaled_memory, memory_profile= memory_profile, b= b, len_memory= len_memory, beta_c = beta_c, beta_cm= beta_cm, beta_g= beta_g, beta_gm= beta_gm)
                        
                        x_inLast2Iterations[0] = copy.deepcopy(x_inLast2Iterations[1])
                        x_inLast2Iterations[1] = copy.deepcopy(x)

                        # print(x)


                  # performance_dict[(initial_condition, memory_profile, b, _lambda , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                  performance_dict[(initial_condition, memory_profile, b, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = (iteration, x_history)
                  # print("performance_dict_initialCondition = ",  performance_dict_initialCondition)
                  
                  # performance_dict[( memory_profile , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = performance_dict_initialCondition


  for memory_profile in memory_profiles:

    if memory_profile in ("fractional_v1", "fractional_v2"):
      for _lambda in _lambdas:
        for len_memory in lens_memory:
          for beta_c in betas_c:
            for beta_cm in betas_cm:
              for beta_g in betas_g:
                for beta_gm in betas_gm:


                  fs_private = [f1, f2]
                  x_history = []

                  x1 = copy.deepcopy(np.array(
                      [ [initial_condition[0]] ,
                        [initial_condition[1]] ]
                      ))
                                        
                  x2 = copy.deepcopy(np.array(
                      [ [initial_condition[0]] ,
                        [initial_condition[1]] ]
                      ))
                  
                  x = [x1, x2]
                  
                  # print("type x1, x2, x: ", type(x1), type(x2), type(x))
                  # print("type contents x1, x2: ", type(x1[0][0]), type(x1[1][0]), type(x2[0][0]), type(x2[1][0]))
                  # print("Shape initial x1, x2 = ", np.shape(x1), np.shape(x2))

                  # print("Shape initial x = ", np.shape(x))
                  # print()
                  # print("Initial x:")
                  
                  # print(x)


                  x_inLast2Iterations = [copy.deepcopy(x), copy.deepcopy(x)]
                  n_agents = len(fs_private)
                  n_params = len(x[0])
                  consensus_memory = np.zeros([n_agents, n_params, len_memory])
                  gradient_memory = np.zeros([n_agents, n_params, len_memory])

                  for iteration in range(max_iterations):

                        last_iteration = last_iteration + 1
                        # np array of shape = (len(x_inLast2Iterations) * n_agents * n_params, )
                              #   contains the absolute difference of each parameter of the agents in the last 2 iterations from the optimum.
                        dif_fromOptimum = np.reshape( [ [ [abs(x_inLast2Iterations[k][i][j,0] - x_opt[j]) for j in range(len(x_inLast2Iterations[0][0]))] for i in range(len(x_inLast2Iterations[0])) ] for k in range(len(x_inLast2Iterations))], newshape= -1)

                        if all(dif < stopping_condition for dif in dif_fromOptimum):
                              
                              x_history.append(copy.deepcopy(x))
                              # performance_dict[(initial_condition, memory_profile , b, _lambda, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                              performance_dict[(initial_condition, memory_profile, _lambda , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] =  (iteration, x_history)

                              break
                        
                        x_history.append(copy.deepcopy(x))

                        x, consensus_memory, gradient_memory = step_withMemory(x, consensus_memory, gradient_memory, fs_private, scaled_memory= scaled_memory, memory_profile= memory_profile, _lambda= _lambda, len_memory= len_memory, beta_c = beta_c, beta_cm= beta_cm, beta_g= beta_g, beta_gm= beta_gm)
                        
                        x_inLast2Iterations[0] = copy.deepcopy(x_inLast2Iterations[1])
                        x_inLast2Iterations[1] = copy.deepcopy(x)

                        # print(x)

                  performance_dict[(initial_condition, memory_profile, _lambda , len_memory, beta_c, beta_cm,  beta_g, beta_gm)] =  (iteration, x_history)
                  

  for memory_profile in memory_profiles:

    if memory_profile in ("constant", "linear"):   # TODO: having 3 blocks to address this problem seems redundant, figure out a way to do it cleaner
      for len_memory in lens_memory:
        for beta_c in betas_c:
          for beta_cm in betas_cm:
            for beta_g in betas_g:
              for beta_gm in betas_gm:
              

                fs_private = [f1, f2]
                x_history = []

                x1 = copy.deepcopy(np.array(
                    [ [initial_condition[0]] ,
                      [initial_condition[1]] ]
                    ))
                                      
                x2 = copy.deepcopy(np.array(
                    [ [initial_condition[0]] ,
                      [initial_condition[1]] ]
                    ))
                
                x = [x1, x2]


                x_inLast2Iterations = [copy.deepcopy(x), copy.deepcopy(x)]
                n_agents = len(fs_private)
                n_params = len(x[0])
                consensus_memory = np.zeros([n_agents, n_params, len_memory])
                gradient_memory = np.zeros([n_agents, n_params, len_memory])

                for iteration in range(max_iterations):

                      last_iteration = last_iteration + 1
                      # np array of shape = (len(x_inLast2Iterations) * n_agents * n_params, )
                            #   contains the absolute difference of each parameter of the agents in the last 2 iterations from the optimum.
                      dif_fromOptimum = np.reshape( [ [ [abs(x_inLast2Iterations[k][i][j,0] - x_opt[j]) for j in range(len(x_inLast2Iterations[0][0]))] for i in range(len(x_inLast2Iterations[0])) ] for k in range(len(x_inLast2Iterations))], newshape= -1)

                      if all(dif < stopping_condition for dif in dif_fromOptimum):
                            
                            x_history.append(copy.deepcopy(x))
                            # performance_dict[(initial_condition, memory_profile , b, _lambda, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = {"n_iterationsUntilConvergence": iteration , "last_x": x, "x_history": np.array(x_history)} 
                            performance_dict[(initial_condition, memory_profile, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = (iteration, x_history)


                            break

                      x_history.append(copy.deepcopy(x))


                      x, consensus_memory, gradient_memory = step_withMemory(x, consensus_memory, gradient_memory, fs_private, scaled_memory= scaled_memory, memory_profile= memory_profile, len_memory= len_memory, beta_c = beta_c, beta_cm= beta_cm, beta_g= beta_g, beta_gm= beta_gm)
                      
                      x_inLast2Iterations[0] = copy.deepcopy(x_inLast2Iterations[1])
                      x_inLast2Iterations[1] = copy.deepcopy(x)

                      # print(x)

                performance_dict[(initial_condition, memory_profile, len_memory, beta_c, beta_cm,  beta_g, beta_gm)] = (iteration, x_history)


  return performance_dict 

































def plot_streamlined(alphas, betas, n_iterations= 100, step_version= "step_sequential"):

    """
    Parameters:
    -----------

    Lists with the hyperparameters of the baseline algorithm (CT approach for DO)

    
    Returns:
    -------
    plot: replication of the convergence curves from "Control approach to distributed optimization" - J Wang

    """

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    

    for alpha in alphas:
      for beta in betas:

            x1 = np.array(         # picking arbitrary initial parameters for the agents's states (2 parameters for a quadratic function) 
                        [ [0.],
                          [0.] ]
                        )

            x2 = np.array(        
                        [ [0.],
                          [0.] ]
                        )

            x3 = np.array(        
                        [ [0.],
                          [0.] ]
                        )

            x4 = np.array(        
                        [ [0.],
                          [0.] ]
                        )


            z1 = np.array(         # z (memory) MUST start at 0
                        [ [0.],
                          [0.] ]
                        )

            z2 = np.array(        
                        [ [0.],
                          [0.] ]
                        )

            z3 = np.array(        
                        [ [0.],
                          [0.] ]
                        )

            z4 = np.array(        
                        [ [0.],
                          [0.] ]
                        )



            # defining private objective functions

            # f1 = lambda x: np.squeeze(x.T @ P[0] @ x + b[0].T @ x + c[0])
            # f2 = lambda x: np.squeeze(x.T @ P[1] @ x + b[1].T @ x + c[1])
            # f3 = lambda x: np.squeeze(x.T @ P[2] @ x + b[2].T @ x + c[2])
            # f4 = lambda x: np.squeeze(x.T @ P[3] @ x + b[3].T @ x + c[3])

            def f1(x1, x2):
               
                import jax.numpy as jnp
                x = jnp.array(         
                            [ [x1],
                              [x2] ]
                            )
                
                return np.squeeze(x.T @ P[0] @ x + b[0].T @ x + c[0])
           
            def f2(x1, x2):
               
                import jax.numpy as jnp
                x = jnp.array(         
                            [ [x1],
                              [x2] ]
                            )
                
                return np.squeeze(x.T @ P[1] @ x + b[1].T @ x + c[1])
            
            def f3(x1, x2):
               
                import jax.numpy as jnp
                x = jnp.array(         
                            [ [x1],
                              [x2] ]
                            )
                
                return np.squeeze(x.T @ P[2] @ x + b[2].T @ x + c[2])
            
            def f4(x1, x2):
               
                import jax.numpy as jnp
                x = jnp.array(         
                            [ [x1],
                              [x2] ]
                            )
                
                return np.squeeze(x.T @ P[3] @ x + b[3].T @ x + c[3])



            x = [x1, x2, x3, x4]
            z = [z1, z2, z3, z4]
            fs_private = [f1, f2, f3, f4]

            history_f_global = []
            history_agent1_parameter1 = []
            history_agent1_parameter2 = []

            history_agent1_memory1 = []
            history_agent1_memory2 = []

            for _ in range(n_iterations):

                # history_f_global.append(f_global(x, fs_private))
                history_agent1_parameter1.append(x[0][0])
                history_agent1_parameter2.append(x[0][1])
                history_agent1_memory1.append(z[0][0])
                history_agent1_memory2.append(z[0][1])

                if step_version == "step_sequential":
                  x, z = step_sequential(x, z, fs_private, alpha= alpha, beta= beta)
                
                if step_version == "step_parallel":
                  x, z = step_parallel(x, z, fs_private, alpha= alpha, beta= beta)


            # ax.plot(range(len(history_f_global)), history_f_global, label= f"alpha={alpha}, beta={beta}, h={h}")
            # ax.set_title(f"Output Global Ojective")
                
            ax.plot(range(len(history_agent1_parameter1)), history_agent1_parameter1, label= f"$x_1^(1)$ - alpha={alpha}, beta={beta}" )
            ax.plot(range(len(history_agent1_parameter2)), history_agent1_parameter2, label= f"$x_2^(1)$ - alpha={alpha}, beta={beta}" )             


    ax.axhline(y= -2.108559, linestyle='dashed', label= r"$x_1^{opt}$")
    ax.axhline(y= -4.551148, linestyle='dashed', label=r"$x_2^{opt}$", color="orange")

    ax.set_title("Parameter Updates (agent 1)")
    ax.set_xlabel("iterations")
    ax.set_ylabel("state trajectory")
    ax.grid(True, linestyle='--', alpha=0.7)  
    ax.legend()
    plt.show()

    # print("x in last iteration = [", history_agent1_parameter1[-1], history_agent1_parameter2[-1], "]")

            # ax.set_title("Memory updates")
            # ax.plot(range(len(history_agent1_memory1)), history_agent1_memory1, label= "$z_1^1$", color= "blue"  )
            # ax.plot(range(len(history_agent1_memory2)), history_agent1_memory2, label= "$z_1^2$", color= "orange" )
            # ax.legend()
            # plt.show()


