import numpy as np
from matplotlib import pyplot as plt
import copy


"""

Module containing helper functions for COMDO.


NOTE:
- the states of every agent must be arrays containing only type float
    
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



def get_gradientVector_Euler(f, x, h= 1e-2):

    """
    Parameters
    -----------
    f: function
        An arbitrary function.
    x: array-like
        An array with the (multi-dimensional) point for which the subgradient vector of f is desired.
    h: float
        An infinitesimally small step along an axis.
    
    Returns
    -------

    numpy array: shape (len(x), 1)
         Gradient vector of f at point x.
            Contains the partial derivatives of f with respect to x_1, ..., x_n.    
    """

    # if type(x[0]) is not float:         # deprecated functionality; TODO: find different way to ensure elements of x are type float
    #     x = [float(x_i) for x_i in x]


    Df = np.zeros((len(x), 1), dtype=int)    # array with the partial deriatives of f w.r.t. x_1, ..., x_n

    for i in range(len(x)):

        x_aux = np.copy(x)
        
        x_aux[i] = x[i] + h

        Df[i] = (f(x_aux)-f(x)) / h          
    
    return Df


def get_gradientVector_Autograd(f, x):     # TODO

    """
    Parameters
    -----------
    f: function
        An arbitrary function.
    x: array-like
        An array with the (multi-dimensional) point for which the subgradient vector of f is desired.
  
        
    Returns
    ------- 
    numpy array: shape (len(x), 1)
        Gradient vector of f at point x.
            Contains the partial derivatives of f with respect to x_1, ..., x_n.    

    """
    
    import jax

    Df = np.zeros((len(x), 1), dtype=int)

    print("x = ", x)
    print("shape x = ", np.shape(x))

    # x1, x2 = x[0, 0], x[1, 0]
    x1, x2 = x[0], x[1]


    f_partial_x1 = jax.grad(f, argnums= 0)
    f_partial_x2 = jax.grad(f, argnums= 1)

    Df[0, 0] = f_partial_x1(x1, x2)
    Df[1, 0] = f_partial_x2(x1, x2)

    # Df[0] = f_partial_x1(x1, x2)
    # Df[1] = f_partial_x2(x1, x2)

    return Df




def step_v2(x, z, fs_private, beta= 0.2, alpha= 3, a= 1, h= 1e-2, subgradient= "euler"):
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
    

    if subgradient == "euler":

      for i in range(len(fs_private)):
          
          x[i] = x[i] + beta * sum( [ a*(x[j] - x[i]) for j in range(n_agents) if j!= i ] ) \
                      + beta * sum( [ a*(z[j] - z[i]) for j in range(n_agents) if j!= i ] ) \
                      - beta * alpha * get_gradientVector_Euler(fs_private[i], x[i], h= h)
          
    if subgradient == "analytic":    # TODO: fix it

      for i in range(len(fs_private)):
          
          x[i] = x[i] + beta * sum( [ a*(x[j] - x[i]) for j in range(n_agents) if j!= i ] ) \
                      + beta * sum( [ a*(z[j] - z[i]) for j in range(n_agents) if j!= i ] ) \
                      - beta * alpha * gradient_vectors_Analytic[i](x[i])
    
    if subgradient == "autograd":
        
      for i in range(len(fs_private)):
          
            x[i] = x[i] + beta * sum( [ a*(x[j] - x[i]) for j in range(n_agents) if j!= i ] ) \
                        + beta * sum( [ a*(z[j] - z[i]) for j in range(n_agents) if j!= i ] ) \
                        - beta * alpha * get_gradientVector_Autograd(fs_private[i], x[i])
        
    return x, z




def step_v3(x, z, fs_private, beta= 0.2, alpha= 3, a= 1, h= 1e-2, subgradient= "euler"):
    """
    Like step, but agent states get updated all at once, not sequentially.

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
    

    if subgradient == "euler":

      x_aux = np.copy(x)      # using a copy of x in optimization, such that parameters get updated all at once, not sequentially

      for i in range(len(fs_private)):
          
          x[i] = x[i] + beta * sum( [ a*(x_aux[j] - x[i]) for j in range(n_agents) if j!= i ] ) \
                      + beta * sum( [ a*(z[j] - z[i]) for j in range(n_agents) if j!= i ] ) \
                      - beta * alpha * get_gradientVector_Euler(fs_private[i], x[i], h= h)
                  
    return x, z





def plot_streamlined(alphas, betas, hs, n_iterations= 100, subgradient= "euler", step_version= "step_v2"):

    """
    Parameters:
    -----------

    Lists with the hyperparameters of the baseline algorithm (CT approach for DO)

    """

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    

    for alpha in alphas:
      for beta in betas:
        for h in hs:

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

            f1 = lambda x: np.squeeze(x.T @ P[0] @ x + b[0].T @ x + c[0])
            f2 = lambda x: np.squeeze(x.T @ P[1] @ x + b[1].T @ x + c[1])
            f3 = lambda x: np.squeeze(x.T @ P[2] @ x + b[2].T @ x + c[2])
            f4 = lambda x: np.squeeze(x.T @ P[3] @ x + b[3].T @ x + c[3])

            x = [x1, x2, x3, x4]
            z = [z1, z2, z3, z4]
            fs_private = [f1, f2, f3, f4]

            history_f_global = []
            history_agent1_parameter1 = []
            history_agent1_parameter2 = []

            history_agent1_memory1 = []
            history_agent1_memory2 = []

            for _ in range(n_iterations):

                history_f_global.append(f_global(x, fs_private))
                history_agent1_parameter1.append(x[0][0])
                history_agent1_parameter2.append(x[0][1])
                history_agent1_memory1.append(z[0][0])
                history_agent1_memory2.append(z[0][1])

                if step_version == "step_v2":
                  x, z = step_v2(x, z, fs_private, alpha= alpha, beta= beta, h= h, subgradient= subgradient)
                
                if step_version == "step_v3":
                  x, z = step_v3(x, z, fs_private, alpha= alpha, beta= beta, h= h, subgradient= subgradient)


            # ax.plot(range(len(history_f_global)), history_f_global, label= f"alpha={alpha}, beta={beta}, h={h}")
            # ax.set_title(f"Output Global Ojective")
                
            ax.plot(range(len(history_agent1_parameter1)), history_agent1_parameter1, label= f"$x_1$ - alpha={alpha}, beta={beta}, h={h}" )
            ax.plot(range(len(history_agent1_parameter2)), history_agent1_parameter2, label= f"$x_2$ - alpha={alpha}, beta={beta}, h={h}" )             


    ax.axhline(y= -2.108559, linestyle='dashed', label= "optimal $x_1$")
    ax.axhline(y= -4.551148, linestyle='dashed', label= "optimal $x_2$", color="orange")

    ax.set_title("Parameter Updates (agent 1)")
    ax.set_xlabel("iterations")
    ax.set_ylabel("state trajectory")

    ax.legend()
    plt.show()

    print("x in last iteration = [", history_agent1_parameter1[-1], history_agent1_parameter2[-1], "]")

            # ax.set_title("Memory updates")
            # ax.plot(range(len(history_agent1_memory1)), history_agent1_memory1, label= "$z_1^1$", color= "blue"  )
            # ax.plot(range(len(history_agent1_memory2)), history_agent1_memory2, label= "$z_1^2$", color= "orange" )
            # ax.legend()
            # plt.show()


