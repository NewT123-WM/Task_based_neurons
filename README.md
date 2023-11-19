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

# Experiment

To further illustrate the necessity and effectiveness of using symbolic regression to generate neurons for different tasks, we perform experiments on 20 public datasets: 10 for classification and 10 for regression.

We first perform the vectorized symbolic regression on the normalized dataset. The relevant information regarding these 20 datasets, and the regression results are shown in Table \ref{tab:symbolic_res}.

\begin{table*}[htb!]
\caption{The formulas learned by the vectorized symbolic regression over 20 public data.}
\vspace{-0.3cm}
    \centering
   \scalebox{0.8}{ \begin{tabular}{lllll}
        \toprule  
Datasets & Instances & Features & Classes & Predicted Function  \\
        \hline 
       
 california housing & 20640 & 8 & continuous & $\bm{0.068}(\x\odot^3\x)^\top+\bm{0.15}\x^\top+0.76$  \\
 house sales & 21613 & 15 & continuous & $-\bm{0.062}(\x\odot^4\x)^\top+\bm{0.025}(\x\odot^3\x)^\top-\bm{0.010}(\x\odot\x)^\top+ \bm{0.067}\x^\top+0.74$   \\
 airfoil self noise & 1503 & 5 & continuous & $\bm{0.064}(\x\odot\x)^\top-\bm{0.038}\x^\top-0.087$   \\
  wine quality & 6497 & 11 & continuous & $\bm{0.0076}(\x\odot^4\x)^\top+\bm{0.055}(\x\odot^3\x)^\top+\bm{0.10}(\x\odot\x)^\top+\bm{0.055}\x^\top-0.00034$   \\
  fifa & 18063 & 5 & continuous & $\bm{0.30}(\x\odot^5\x)^\top-\bm{0.63}(\x\odot^4\x)^\top-\bm{0.10}(\x\odot^3\x)^\top+\bm{0.38}(\x\odot\x)^\top+\bm{0.13}\x^\top+0.010$   \\
  diamonds & 53940 & 9 & continuous & $-\bm{0.075}(\x\odot^7\x)^\top+\bm{0.16}(\x\odot^6\x)^\top+\bm{0.10}(\x\odot^5\x)^\top-\bm{0.27}(\x\odot^4\x)^\top+\bm{0.090}(\x\odot^3\x)^\top$   \\
  abalone & 4177 & 8 & continuous & $-\bm{0.088}(\x\odot^3\x)^\top-\bm{0.12}(\x\odot\x)^\top+\bm{0.046}\x^\top$   \\
  Bike Sharing Demand & 17379 & 12 & continuous & $-\bm{0.081}(\x\odot\x)^\top+\bm{0.054}\x^\top$   \\
  space ga & 3107 & 6 & continuous & $\bm{0.052}(\x\odot^4\x)^\top+\bm{0.12}(\x\odot^3\x)^\top+\bm{0.025}(\x\odot\x)^\top-\bm{0.073}\x^\top+0.54$   \\
  Airlines DepDelay & 8000 & 5 & continuous & $\bm{0.010}(\x\odot\x)^\top+\bm{0.042}\x^\top-0.27$  \\ \hline \hline
  credit & 16714 & 10 & 2 & $-\bm{0.43}(\x\odot^4\x)^\top+\bm{0.37}(\x\odot\x)^\top+0.21$  \\
  heloc & 10000 & 22 & 2 & $\bm{0.031}(\x\odot^6\x)^\top-\bm{0.026}(\x\odot^5\x)^\top+\bm{0.055}(\x\odot^3\x)^\top$   \\
  electricity & 38474 & 8 & 2 & $-\bm{0.21}(\x\odot\x)^\top+\bm{0.21}\x^\top+1.18$   \\
  phoneme & 3172 & 5 & 2 & $\bm{1.36}(\x\odot^5\x)^\top-\bm{2.91}(\x\odot^3\x)^\top+\bm{0.60}(\x\odot\x)^\top+\bm{1.22}\x^\top$   \\
  bank-marketing & 10578 & 7 & 2 & $-\bm{1.04}(\x\odot^4\x)^\top-\bm{0.14}(\x\odot^3\x)^\top+\bm{0.81}(\x\odot\x)^\top+\bm{0.043}\x^\top-0.068$  \\
 MagicTelescope & 13376 & 10 & 2 & $-\bm{0.30}(\x\odot^4\x)^\top+\bm{1.13}(\x\odot^3\x)^\top+\bm{0.46}(\x\odot\x)^\top-\bm{0.060}\x^\top+1.02$   \\
 vehicle & 846 & 18 & 4 & $-\bm{0.074}(\x\odot^4\x)^\top+\bm{0.068}(\x\odot^3\x)^\top+\bm{0.072}(\x\odot\x)^\top+\bm{0.0015}\x^\top$  \\
 Oranges-vs.-Grapefruit & 10000 & 5 & 2 & $-\bm{0.52}(\x\odot\x)^\top+\bm{0.70}\x^\top+0.89$   \\
 eye movements & 7608 & 20 & 2 & $-\bm{0.017}(\x\odot^3\x)^\top-\bm{0.011}(\x\odot\x)^\top$   \\
contaminant & 2400 & 30 & 2 & $-\bm{0.75}(\x\odot^5\x)^\top-\bm{0.67}(\x\odot^4\x)^\top+\bm{0.38}(\x\odot^3\x)^\top+\bm{0.24}(\x\odot\x)^\top+\bm{0.13}\x^\top$   \\
        \bottomrule 
    \end{tabular}   } 
    \label{tab:symbolic_res}
    \vspace{-0.6cm}
\end{table*}
