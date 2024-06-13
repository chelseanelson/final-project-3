## Scripts

### Documents 

`1a_initial_setup.R`: R script containing the initial splitting of data as well as exploration of the target variable 

`1b_eda.R`: R script containing the univariate and bivariate exploration of variables 

`2_recipes.R`: R script containing the recipes used during model testing

`3_fit_nbayes.R`: R script containing the fitting of resamples for the naive bayes model

`3_tune_enet.R`: R script containing the tuning of the elastic net model

`3_tune_ensemble.R`: R script containing the tuning of the ensemble model

`3_tune_knn.R`: R script containing the tuning of the knn model

`3_tune_mars.R`: R script containing the tuning of the mars model

`3_tune_nnet.R`: R script containing the tuning of the neural network model

`3_tune_rf.R`: R script containing the tuning of the random forest model

`3_tune_svm_poly.R`: R script containing the tuning of the svm polynomial model

`3_svm_rad.R`: R script containing the tuning of the svm radial model

`4_model_analysis.R`: R script containing the comparison analysis of the first round of model testing

`5_tune_improve_nnet.R`: R script containing the tuning of the improved neural network model

`5_tune_improve_rf.R`: R script containing the tuning of the improved rf model

`5_tune_improve_svm_poly.R`: R script containing the tuning of the improved svm polynomial model

-`6_improve_model_analysis.R`: R script containing the comparison analysis of our improved round of model testing

`7_train_final_model.R`: R script containing the training of the final best model on the full training set

`8_assess_final_model.R`: R script containing the assessment of the final best model on the full testing set

### Folders

`ensemble/`: contains the tuning and assessment of ensemble model 

`var-selection/`: contains the creation of lasso-based variable selection for improved recipe

### Usage

If you want to explore or run the code, we would recommend starting with `rscripts/1a_initial_setup.R` to understand the distribution and layout of the original data and then running in order after that in order to see the development and creation of the different types of models. We would finish off by looking at `rscripts/4_model_analysis.R` through `rscripts/8_assess_final_model.R` to see how each model performs, looking at which has the best performance under ROC_AUC. Otherwise, all of the associated model data, fitted data, tables of the performance analyses, and the recipes in which we created can be found in their respective rscripts.