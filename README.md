# IEEE-Fraud Kaggle Challenge with CatBoost

AKA submitting a solution to Kaggle for the first time, on a Saturday meetup.

The main goal of this was to merge the raw data, use only data types immediately usable in the model, generate a list of propensities for fraud on each transaction from the model and submission. I extended it with a graphical finish by including the SHAP library, which plays well with Catboost. 

## Useful Links

* [Challenge and dataset on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection)

* [CatBoost Paribas tutorial](https://github.com/catboost/tutorials/blob/master/competition_examples/kaggle_paribas.ipynb) - very similar to IEEE dataset, also highly skewed  

* [CatBoost SHAP visualisation library tutorial](https://github.com/catboost/tutorials/blob/master/model_analysis/shap_values_tutorial.ipynb) new addition by the CatBoost team since I last tried it on this submission. Lets you visualise features that contributed the most to model results.

* [SHAP library](https://github.com/slundberg/shap)

## Getting Started

The whole project can be followed on the [Fraud CatBoost Notebook](https://github.com/NatMota/kaggle-ieee-fraud/blob/master/Fraud%20Catboost.ipynb) using Python.

### Prerequisites

Packages

```
pandas
numpy
itertools
catboost
shap
```

This was trained on a Google Cloud N2 machine with 8 cores 16GB ram, which is the most powerfull VM you can run within the free $300 credits. Creating it as a Windows VM is more expensive but is easiest to interface with coming from a Windows user, for Anaconda's Python setup and training data transfer with Google Drive.

With the raw dataset and minimal manipulation it takes 30 minutes to train on the setup above. This can be improved with feature selection, variable downcasting and converting back to numpy arrays once you're happy with the inital shaping. The raw joined data is really large and takes a lot of memory to train on.

## Notes

This method got 90% accuracy on Kaggle - it missed 10% of frauds so to say. As of June 2020 this is on the top 66% of the leaderboard, which means most of the work went into optimizing the remaining 8% - and the top submission had AUC of 94.6%. So it's a good result with minimal configuration.

The features are anonymized for the most part with some efforts on the Kaggle forums revealing part of them. Having in mind we only looked at numerical features, the shap_values plot shows that the transaction amount was an example of a useful predictor for this model. As mentioned by the creators of the library, it might not show a direct causal explanation but graphically showing what the model is focusing on is a welcome addition to the pipeline. 

Extensions:

* Feature testing
* Feature engineering
* Dataset enrichment from other sources
* Comparison with other boosting models
* Bagging
* Scaling sampling for balanced dataset with k-folds averaging
