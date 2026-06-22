# Liquid Neural Networks for Smartphone-Based Human Activity Recognition
#### *Comparing the accuracy, parameter efficiency, and temporal perturbation robustness of continuous-time models with CfC and GRU on the UCI HAR sequence classification task.*


* CfC is a continuous-time model derived from the approximate closed-form solution of LTC dynamics, where time explicitly influences its state updates.
* The official ncps implementation directly provides PyTorch's CfC and LTC, with the input format being (batch, time, features).
* One of the design goals of CfC is to preserve continuous-time modeling while avoiding the computational bottleneck of general ODE solvers.

## Q1 : What are the core assumptions that CfC and traditional GRU differ in when processing time series data?




## Q2 : How can UCI HAR data be transformed into a sequence classification problem suitable for a recurrent model?




## Q3: Can CfC successfully identify six human activities from sensor sequences?




## Q4: Under similar experimental conditions, what are the advantages and disadvantages of CfC compared to GRU?



## Q5: When the time series is incomplete, is CfC more robust than GRU?