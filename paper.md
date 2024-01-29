---
title: 'Pycausal-Explorer: A Scikit-learn compatible causal inference toolkit'
tags:
  - Python
  - causal inference
  - scijit-learn
  - machine learning
authors:
  - name: Guilherme Goto Escudero
    orcid: 0009-0000-2451-7029
    equal-contrib: true
    corresponding: true
    affiliation: 1
  - name: Heitor de Moraes Santos
    affiliation: 2
  - name: João Vitor Tigre Almeida
    affiliation: 2
  - name: Roseli de Deus Lopes
    affiliation: 1
affiliations:
 - name: Escola Politécnica - Universidade de São Paulo, Brazil
   index: 1
 - name: Independent Researcher, Brazil
   index: 2
date: 26 January 2024
bibliography: paper.bib

---

# Summary

Pycausal-Explorer is an open source, scikit-learn compatible Python library which 
leverages causal inference and machine learning for causal reasoning and exploration. 
It consists of a number of algorithms built to calculate treatment effects and methods 
concerning causal analysis. It has the foundation to allow researchers to easily study 
and compare different causal inference algorithms. Pycausal-Explorer is a modular 
package with a common scikit-learn-like interface, reducing the gap from research to 
production environments.

# Statement of need

Causal inference has been an ascending research topic over the last years. Some recent 
work conjugate machine learning and causal inference in order to estimate treatment 
effect from observational data, some examples being the use of meta-learners [@kunzel2019metalearners] 
and honest causal trees [@wager2018estimation]. 

In such a landscape, `Pycausal-Explorer` is a Python package that helps create, 
compare and publish methods which estimate treatment effects from observational data. 
The package has two main objectives: (1) leverage the study of causal inference 
through a common framework, and (2) close the gap between research of the field and 
real-world applications. The library is an open source project built in Python, 
providing algorithms, methods, and data sets commonly found in the causal inference 
literature. `Pycausal-Explorer` is compatible with scikit-learn [@scikit-learn], 
which expands such a well-known library for building and monitoring causal models. 
It has been extensively tested and has documented guidelines so that contributions 
can be made to the library. 

The library implements a Python class called BaseCausalModel from which all models 
should inherit. It is an abstract class inherited from scikit-learn's BaseEstimator. 
All models must implement the following methods: fit, predict, estimate_ate and 
estimate_ite. Whenever a model is not capable of measuring individual treatment 
effect, individual treatment effects will be defined as the average treatment effect, 
regardless of the individual's particularities. The models currently implemented 
include linear regression and inverse propensity score weighting (IPTW) for average 
treatment effect and k-nearest neighbours, S-learner, T-Learner, X-Learner, RA-Learner,  
DR-Learner, DoubleML and a novel method called randomized trees ensemble embedding to 
calculate heterogeneous treatment effect.

Though `Pycausal-Explorer` is unique in that it enables causal model exploration in 
the form of a scikit-learn compatible Python tool, there are other prominent toolkits 
aimed at causal inference which also bring a lot of value to research and business 
applications.

Causal explorer [@aliferis2003causal] is a MATLAB package with causal inference tools 
meant for biomedical applications, and it is perhaps one of the first published libraries 
aimed at working with causality. It provides tools to model causal relations between 
variables as causal probabilistic networks (which are bayesian networks) and a handful of 
causal discovery algorithms which help choose the best causal graph assumption.

Generalized random forests [@athey2019generalized] is an open source R package which 
implements causal models based on tree ensembles, supporting the so-called honest 
estimation of causal effects. Although it is a very popular language in academics, 
specially in the econometrics field, Python tends to be more used than R in business 
applications and is easier to integrate with APIs and external software, so there might 
be demand for a Python implementation of generalized random forests.

Furthermore, there are also examples of business-orientated Python libraries for causal 
modeling. CausalML [@chen2020causalml] is a library made by Uber which is focused 
on uplift models (i.e., estimating effects of interventions in scenarios such as A/B 
tests). EconML [@econml] is a library made by Microsoft which leverages machine 
learning tools to estimate treatment effects, particularly for applications in Economics. 
Another library by Microsoft called DoWhy [@dowhy] implements an end-to-end causal 
analysis tool, abstracting most of the data process work. This last library focuses on 
identification of causal effect (i.e., inspecting causal models’ assumptions) and allows 
for integration with algorithms implemented by both CausalML and EconML.

# References
