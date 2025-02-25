# FrODO: Fractional Order Distributed Optimization
(talk: https://youtu.be/f_yNoEghRhg?si=UxbdxmTKy5GAVKuo)

(paper: https://arxiv.org/abs/2412.02546)


Buiding an ML optimizer with competitve performance by integrating a computational neuroscience model into stochastic gradient descent.

The implementation of the algorithm for the illHess and Rosenbrock settings can be found in `FOLDER_code/comdo/utils_proper.py`. For ANNs, the implementation of FrODO and auxiliary function for reproducible results are in `FOLDER_code/comdo/utils_ANNs.py` and a training script in `FOLDER_code/test_ANNs.ipynb`.

The code used for tuning the hyperparameters of FrODO can be found in `FOLDER_code/tuning_Lambdas.ipynb`. The results of the illHess and Rosen experiments can be reproduced with the initial conditions from `initial_conditions_illHess` and `initial_conditions_Rosen`.

Main outcomes of the project:

### Achieved stability across objectives with ill-defined Hessian matrices
![image](https://github.com/AndreiLix/FrODO/assets/94043928/e54c963f-bc52-49a8-9397-c190bcc62b61)


### Achieved competitive performance with state of the art when scaling up to Federated Learning 
![image](https://github.com/user-attachments/assets/c28ed992-8127-4def-b318-f2e3881d2992)

![image](https://github.com/AndreiLix/FrODO/assets/94043928/e668dc2e-0fa7-401f-a09a-d51c5f843f1f)
