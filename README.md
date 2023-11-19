# No One-Size-Fits-All Neurons: Task-based Neurons for Artificial Neural Networks
This is the python implementation (PyTorch) of using nonlinear neurons to build a neural network.

In this work,
+ Different from traditional research focusing on the design of neural network architecture, we focus on the optimization of neurons. we propose a two-step framework for prototyping task-based neurons. 
+ First, we utilize vectorized symbolic regression to identify the optimal formula that fits the input data. Secondly, we parameterize the obtained formula and use it as neurons to build a neural network. This framework organically combines symbolism and connectionism, giving full play to the advantages of both, thereby improving the fitting ability of artificial neural networks.
+ We illustrate the flexibility, necessity, and effectiveness of our proposed framework from multiple perspectives. We tested on synthetic data, public datasets, and real-world data. The test results show that the fitting ability of the network built using nonlinear neurons is better than that of the traditional linear neural network.
