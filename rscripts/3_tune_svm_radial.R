# ACCS Final Project ----
# Define and fit SVM Radial model
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(tictoc)

# handle common conflicts 
tidymodels_prefer()

# set up parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load folded data 
load(here("data-splitting/hotel_folds.rda"))

# load pre-processing/feature engineering recipe 
load(here("recipes/2_recipe_para.rda"))

# model specifications ----
svm_radial_model <- 
  svm_rbf(
    mode = "classification",
    cost = tune(),
    rbf_sigma = tune()
  ) %>%
  set_engine("kernlab")

# define workflows 
svm_radial_wflow <-
  workflow() %>%
  add_model(svm_radial_model) %>%
  add_recipe(recipe_para)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(svm_radial_model)

# change hyperparameter ranges 
svm_radial_params <- extract_parameter_set_dials(svm_radial_model)

# build tuning grid 
svm_radial_grid <- grid_latin_hypercube(svm_radial_params, size = 30)

# tune workflows/models ----
# set seed
set.seed(02848)
tic.clearlog() # clear log
tic("svm_radial") # start clock

tuned_svm_radial <-
  svm_radial_wflow %>%
  tune_grid(
    hotel_folds,
    grid = svm_radial_grid,
    control = control_grid(save_workflow = TRUE)
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_radial <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows)
save(tuned_svm_radial, tictoc_svm_radial,
     file = here("results/fitted_tuned_models/tuned_svm_radial.rda"))