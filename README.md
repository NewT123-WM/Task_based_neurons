# No One-Size-Fits-All Neurons: Task-based Neurons for Artificial Neural Networks
This is the python implementation (PyTorch) of using nonlinear neurons to build a neural network.

In this work,
+ Different from traditional research focusing on the design of neural network architecture, we focus on the optimization of neurons. we propose a two-step framework for prototyping task-based neurons. 
+ First, we utilize vectorized symbolic regression to identify the optimal formula that fits the input data. Secondly, we parameterize the obtained formula and use it as neurons to build a neural network. This framework organically combines symbolism and connectionism, giving full play to the advantages of both, thereby improving the fitting ability of artificial neural networks.
+ We illustrate the flexibility, necessity, and effectiveness of our proposed framework from multiple perspectives. We tested on synthetic data, public datasets, and real-world data. The test results show that the fitting ability of the network built using nonlinear neurons is better than that of the traditional linear neural network.

# Abstract
In the past decade, many successful networks are on novel architectures, which almost exclusively use the same type of
neurons. Recently, more and more deep learning studies have been inspired by the idea of NeuroAI and the neuronal diversity
observed in human brains, leading to the proposal of novel artificial neuron designs. Designing well-performing neurons represents a new dimension relative to designing well-performing neural architectures. Biologically, the brain does not rely on a single type of neuron that universally functions in all aspects. Instead, it acts as a sophisticated designer of task-based neurons. In this study, we address the following question: since the human brain is a task-based neuron user, can the artificial network design go from the task-based architecture design to the task-based neuron design? Since methodologically there are no one-size-fits-all neurons, given the same structure, task-based neurons can enhance the feature representation ability relative to the existing universal neurons due to the intrinsic inductive bias for the task. Specifically, we propose a two-step framework for prototyping task-based neurons. First, symbolic regression is used to identify optimal formulas that fit input data by utilizing base functions such as logarithmic, trigonometric, and exponential functions. We introduce vectorized symbolic regression that stacks all variables in a vector and regularizes each input variable to perform the same computation, which can expedite the regression speed, facilitate parallel computation, and avoid overfitting. Second, we parameterize the acquired elementary formula to make parameters learnable, which serves as the aggregation function of the neuron. The activation functions such as ReLU and the sigmoidal functions remain the same because they have proven to be good. As the initial step, we evaluate the proposed framework via systematic experiments on tabular data and using polynomials as base functions. Empirically, experimental results on synthetic data, classic benchmarks, and real-world applications show that the proposed task-based neuron design is not only feasible but also delivers competitive performance over other state-of-the-art models. 

# Two-step framework
![image](https://github.com/NewT123-WM/Task_based_neurons/blob/main/framework.png）
