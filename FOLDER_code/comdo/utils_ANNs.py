import jax.numpy as jnp
from matplotlib import pyplot as plt
import copy
import sklearn
import torch
import numpy as np
import equinox as eqx



def get_2DO_datasets(train_dataset, test_dataset, BATCH_SIZE= 64):
    """
    Parameters:
    ----------
    train_dataset: torch.data.utils.data.Dataset
    test_dataset: torch.data.utils.data.Dataset
    
    Retruns:
    --------
    Agent1_Train_dataset, Agent1_Test_dataset, Agent2_Train_dataset, Agent2_Test_dataset
        Input dataset split into 2 distinct, balanced datasets. (through data elimination)
    
    """
    train_data = np.array(train_dataset.data)
    train_targets = np.array(train_dataset.targets)

    balanced_train_data = []
    balanced_train_targets = []
    count_unique_targets = len(np.unique(train_targets))
    counts = count_unique_targets * [0]            # TODO: replace 10 with the 

    # count of the least frequent target
    count_least_freq_train = min(np.bincount(train_targets))


    for i, target in enumerate(train_targets):
        if counts == count_unique_targets * [count_least_freq_train]:
            break
        
        if counts[target] < count_least_freq_train:
            counts[target] += 1
            balanced_train_targets.append(target)
            balanced_train_data.append(train_data[i])

        
    balanced_train_targets = np.array(balanced_train_targets)
    balanced_train_data = np.array(balanced_train_data)

    test_data = np.array(test_dataset.data)
    test_targets = np.array(test_dataset.targets)

    balanced_test_data = []
    balanced_test_targets = []
    counts = count_unique_targets * [0]


    # count of the least frequent target
    count_least_freq_test = min(np.bincount(train_targets))

    for i, target in enumerate(test_targets):
        if counts == count_unique_targets * [count_least_freq_test]:
            break

        if counts[target] < count_least_freq_test:
            counts[target] += 1
            balanced_test_targets.append(target)
            balanced_test_data.append(test_data[i])

    balanced_test_targets = np.array(balanced_test_targets)
    balanced_test_data = np.array(balanced_test_data)

    X1_train, X2_train, y1_train, y2_train = sklearn.model_selection.train_test_split(balanced_train_data,balanced_train_targets, train_size = 0.5, stratify=balanced_train_targets)
    X1_test, X2_test, y1_test, y2_test = sklearn.model_selection.train_test_split(balanced_test_data,balanced_test_targets, test_size = 0.5, stratify=balanced_test_targets)


    class CustomDataset(torch.utils.data.Dataset):

        import numpy as np

        def __init__(self, data, targets):
            data = np.array(data, dtype= np.float32)
            data = np.expand_dims(data, axis= 0)
            self.data = torch.from_numpy(data)

            targets = np.array(targets)
            self.targets = torch.from_numpy(targets) 

        def __len__(self):
            return len(self.targets)

        def __getitem__(self, idx):

            return self.data[:, idx, :, :], self.targets[idx]


    Agent1_Train_dataset = CustomDataset(
        data= X1_train,
        targets= y1_train
    )
    Agent1_Test_dataset = CustomDataset(
        data= X1_test,
        targets= y1_test
    )

    Agent2_Train_dataset = CustomDataset(
        data= X2_train,
        targets= y2_train
    )
    Agent2_Test_dataset = CustomDataset(
        data= X2_test,
        targets= y2_test
    )


    return Agent1_Train_dataset, Agent1_Test_dataset, Agent2_Train_dataset, Agent2_Test_dataset



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

    def __init__(self, models, len_memory= 100, _lambda= 0.15, beta_c = 0.2, beta_cm= 0.04, beta_g= 0.6, beta_gm= 0.36):
        
        """
        Parameters:
        ------------

        self.models : list
            List with the PyTrees of the models in the DO network.

        self.len_memory : int 
            Memory length.

        self._lambda : float in (0,1)
            The fractional order to use forte fracitonal memory weighting.

        self.beta_c : flaot
            Weight of the consensud terms.
                Default beta_c = beta_g

        self.beta_g : float
            Weight of the private gradient descent term.

        self.beta_gm : float
            Weight of the gradient memory feedback term.
        
      Returns
      --------
      list : shape (n_agents, <parameters>)
        List containing the agents and their updated parameters.
        """

        self.models = models
        self._lambda = _lambda
        self.beta_c = beta_c
        self.beta_g = beta_g
        self.beta_gm = beta_gm


        self.n_agents = len(models)
        self.len_memory = len_memory
        self.idx_layersWithWeights = []
        self.n_layers = len(self.models[0].layers)
        self.gradient_memory = self.n_agents * [ self.len_memory * [{}] ]
        self.z_g = self.n_agents * [{}] 

        for i in range(len(models[0].layers)):     
            if hasattr(models[0].layers[i], "weight"):
                self.idx_layersWithWeights.append(i)

        for layer_i in self.idx_layersWithWeights:
            # for dim in np.shape(models[0].layers[i].weight):  # poisoned
            for agent_i in range(self.n_agents):

                self.z_g[agent_i][layer_i] = np.zeros( np.shape(models[0].layers[layer_i].weight) )
                for memory_i in range(len_memory):
                    self.gradient_memory[agent_i][memory_i][layer_i] = np.zeros( np.shape(models[0].layers[layer_i].weight)  )

        # print("Shape z_g =")
        # print(self.shape_z_g)



    def step_withMemory(self, models, grads_list):
        
        """

        Changes the states (x), consensus_memory and gradient_memory of our agents based on a step in the optimization process.

        Parameters
        ----------
        x: np array, shape (n_agents, n_params, 1).
            Agent states.

        gradient_memory: np array, shape (n_agents, self.len_memory, n_layers, <layer shape>)
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

        x = models
        n_agents = self.n_agents

        #---------------------------- first stage --------------------------------------

        # computing the terms used for the updates in this iteration


        x = copy.deepcopy(models)

        consesus_term = copy.deepcopy(self.z_g)
        gradient_term = copy.deepcopy(self.z_g)


        for agent_i in range(n_agents):
            for layer_i in self.idx_layersWithWeights:

               gradient_term[agent_i][layer_i] = grads_list[agent_i].layers[layer_i].weight 

        
        for agent_i in range(n_agents):
            for layer_i in self.idx_layersWithWeights:
                consesus_term[agent_i][layer_i] = self.beta_c * sum( [ (x[agent_j].layers[layer_i].weight - x[agent_i].layers[layer_i].weight) for agent_j in range(n_agents) if agent_j!= agent_i ] )

        memory_weights = np.array([fractional_v2(x, self.len_memory, self._lambda) for x in range(self.len_memory)])
        # scaling values between 0 and 1
        memory_weights = memory_weights / max(memory_weights)


        # z_g = np.zeros(self.shape_z_g)

        for agent_i in range(n_agents):
            for layer_i in self.idx_layersWithWeights:

                aux_tensor = np.array([ memory_weights[memory_i] * self.gradient_memory[agent_i][memory_i][layer_i] for memory_i in range(self.len_memory)])

                # print( f"Shape aux_tensor at layer: {layer_i}:", np.shape(aux_tensor) ) 
                
                self.z_g[agent_i][layer_i] = np.sum(aux_tensor, axis= 0)  # summing over the memory axis



        # ================================================================


        #---------------------------- second stage --------------------------------------

        # updating x, consensus_memory and gradient_memory



        # NOTE: all terms in update must be shape (n_agents, n_params, 1)

        for agent_i in range(n_agents):
            for layer_i in self.idx_layersWithWeights:

                # print(f"Shape Agent for layer {layer_i}: ", np.shape(x[agent_i].layers[layer_i].weight))
                # print(f"Shape consensus term for layer {layer_i}: ", np.shape(consesus_term[agent_i][layer_i]))
                # print(f"Shape gradient term for layer {layer_i}: ", np.shape(gradient_term[agent_i][layer_i]))
                # print(f"Shape z_g term for layer {layer_i}: ", np.shape(self.z_g[agent_i][layer_i]))



                # x[agent_i].layers[layer_i].weight =  x[agent_i].layers[layer_i].weight \
                #                             + self.beta_c * consesus_term[agent_i][layer_i] \
                #                             - self.beta_g * gradient_term[agent_i][layer_i] \
                #                             - self.beta_gm * self.z_g[agent_i][layer_i]
                


                aux_update =  x[agent_i].layers[layer_i].weight \
                                + self.beta_c * consesus_term[agent_i][layer_i] \
                                - self.beta_g * gradient_term[agent_i][layer_i] \
                                - self.beta_gm * self.z_g[agent_i][layer_i]
    
                where = lambda m: m[agent_i].layers[layer_i].weight
                x = eqx.tree_at(where, x, aux_update)


                # updating consensus mamory and gradient based on this iteration's sonsensus term and gradient descent term

                self.gradient_memory[agent_i][:-1][layer_i] = self.gradient_memory[agent_i][1:][layer_i]
                self.gradient_memory[agent_i][-1][layer_i] = gradient_term[agent_i][layer_i]


        return x
