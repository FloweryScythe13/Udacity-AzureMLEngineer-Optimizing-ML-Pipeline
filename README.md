# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary

This is a dataset of demographic, financial, and other related attributes for consumers of banking services. For each consumer, it records characteristics such as their age, occupation, marital status, highest attained level of education, whether they have received a housing loan, whether they are recipients of a personal loan, whether they have credit in default, and the number of employees at their bank of choice, among others. According to the Center for Machine Learning and Intelligence Systems at University of California Irving (which hosts the original public dataset), the data herein is "related with direct marketing campaigns (phone calls) of a Portuguese banking institution" ([Source](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)). Our goal in using this dataset is to predict whether a given client of the bank has subscribed or will subscribe to a term deposit (the label column y). 

Based on my solution output, the best performing model was generated using AutoML and utilized a VotingEnsemble algorithm. This model had an accuracy of 91.74%, which beat out the optimized Scikit-learn logistic regression model's accuracy of 91.53%.  
## Scikit-learn Pipeline


My Scikit-learn pipeline used the following steps: 
1. A training script `train.py` which takes in two parameters: a regularization strength `C` and an upper limit on training iterations `max_iter`. 
2. An input dataset, which is the bankmarketing dataset mentioned above. The script fetches the bankmarketing dataset and cleans it automatically. 
3. A logistic regression classifier from Scikit-learn, into which the dataset is fed for training. 
4. A primary metric for scoring the trained model - we chose accuracy as that metric.
2. A random parameter sampler for selecting values from the parameter search space; it draws regularization values from a uniform distribution between 0.25 and 1.75, and pulls `max_iter` values from a discrete set of choices between 60 and 140. 
3. An early termination policy that uses [median stopping](https://learn.microsoft.com/en-us/azure/machine-learning/v1/how-to-tune-hyperparameters-v1#median-stopping-policy), which is based on a running average of the primary metric across job runs. 
4. A HyperDriveRun, which encapsulates the above components in its configuration. 


**Choice of Parameter Sampler**

The RandomParameterSampler is ideal in situations where you do not have a prior history of runs for your dataset and model, and you want to get a good baseline scan of how hyperparameter values compare within a given range - especially if you or your team is on a budget for using your compute resources. I chose this as my parameter sampling approach because all three of those conditions were met in this project. 


**Choice of Early Termination Policy**

According to Golovin et al. ([Google Vizier: A Service for Black-Box Optimization](https://research.google.com/pubs/pub46180.html)), whose findings inspired it, a median stopping policy is model-free in the same way as bandit policies, making it widely applicable for performance curve evaluation. (They define a *performance curve* as comprised of performance & accuracy information gleaned from a trial while that trial is running - for example, the accuracy metrics made available at the end of each training epoch in the trial). In their empirical tests, they found that "in all cases the stopping rule consistently achieved a factor two to three speedup over random search, while always finding the best performing Trial." They also found that "the algorithm almost never decided to stop the ultimately-best-performing trial early."

If one were to use additional parameters for the MedianStoppingPolicy (which I opted to forego), one could look to the suggestion in Microsoft's official documentation to use  `evaluation_interval=1` and `delay_evaluation=5`. These baseline settings "can provide approximately 25%-35% savings with no loss on primary metric" (Microsoft, [MedianStoppingPolicy Class](https://learn.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.medianstoppingpolicy?view=azure-ml-py#remarks)). 

## AutoML Pipeline ##

In contrast to the LogisticRegression algorithm we chose above, the Azure AutoML service found that a VotingEnsemble algorithm produced the most accurate model from child trials. 

## Pipeline comparison


<table>
<th>Model</th><th>Accuracy</th><th>Architecture</th>
<tr>
    <td>LogisticRegression</td><td>0.9153</td><td>Input data + customized model training script + random search parameter sampler + median stopping policy + HyperDriveRun
</tr>
<tr>
    <td>VotingEnsemble</td><td>0.9174</td><td>Input data (with additional pre-processing step) + AutoMLRun</td>
</tr>
</table>
Overall, the VotingEnsemble model was produced from a simpler architecture while at the same time performing as well as the optimized LogisticRegression model (a difference in accuracy of only 0.0021 is not statistically significant). This difference is rooted in the AutoML technology, which democratizes machine learning and reduces the barrier to entry by automating away many of the implementation tasks that could only be competently done by experienced data scientists in years past. No wonder people say that many data scientists will eventually be replaced by AutoML!


## Future work


1. Use non-default values for the median stopping policy, to improve the compute costs for training the Scikit-learn model. 
2. Alternatively, choose a different early termination policy, which could futher improve the trials and allow for finding a better-performing model. 
3. Increase the timeout duration for the AutoMLConfig, to allow the AutoML engine to trial a larger number of algorithms and potentially find an even better training algorithm than VotingEnsemble. 


