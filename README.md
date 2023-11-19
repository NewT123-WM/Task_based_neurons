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
![image](https://github.com/NewT123-WM/Task_based_neurons/blob/main/framework.png)
+ First, we introduce vectorized symbolic regression to construct an elementary neuronal model (Figure a). Symbolic regression draws inspiration from scientific discoveries in physics, aiming to identify optimal formulas that fit input data by utilizing base functions such as logarithmic, trigonometric, and exponential functions. The vectorized symbolic regression stacks all variables in a vector and regularizes each input variable to perform the same computation. Given the complexity and unclear nonlinearity of the tasks, formulas learned from vectorized symbolic regression can capture the underlying patterns in the data, and these patterns are different in different contexts. Thus, fixed formulas used in pre-designed neurons are disadvantageous.
+ Second, we parameterize the acquired elementary formula to make parameters learnable (Figure b), which serves as the aggregation function of the neuron. The role of the vectorized symbolic regression is to identify the basic patterns behind data, the parameterization allows the task-based neurons to adapt and interact with each other within a network.

# Main results

We first perform the vectorized symbolic regression on the normalized dataset. The relevant information regarding these 20 datasets, and the regression results are shown in Table 5.
![image](https://github.com/NewT123-WM/Task_based_neurons/blob/main/table5.png)

Second, we test the superiority of task-based neurons relative to linear ones. We use the same 20 datasets in the last subsection: 10 for regression and 10 for classification. We don't need to repeat the process of the vectorized symbolic regression. Instead, we directly use polynomials learned in Table 5. The training and test sets are divided according to the ratio of $8:2$. For *TN*  and *LN*, the data division and the batch size are the same. We select 5 different network structures for each dataset for a comprehensive comparison. When designing the network structures of *TN*, we ensure that the number of parameters of *TN* is fewer than the *LN* to show the superiority of task-based neurons in efficiency. Each dataset is tested 10 times for reliability of results. The MSE and classification accuracy are presented in the form of $\mathrm{mean}~(\mathrm{std})$ in Table 6.

![image](https://github.com/NewT123-WM/Task_based_neurons/blob/main/table6.png)

# Dataset
Datasets are collected from the scikit-learn package and the official website of OpenML.

# Requirements
h5py

sympy

numpy

pandas  

scikit-learn = 1.2.2

pytorch = 1.13.1

# How to Use
Here we provide a set of synthetic data for display because the synthetic data has real expressions for inspection and comparison.
+ Get a symbolic regression expression.

  ```
  python regression.py
  ```

After getting the neuron expression, you can assign the corresponding symbolic expression to the `expr` variable in `train.py`.

+ Perform network training.

  ```
  python train_set_1.py
  ```

# Citation

If you have any questions about this work, feel free to contract us: wangmeng22@stu.hit.edu.cn

