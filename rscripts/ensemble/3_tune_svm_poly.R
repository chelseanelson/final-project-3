# ACCS Final Project ----
# Define and fit SVM Polynomial model
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)
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
svm_poly_model <- 
  svm_poly(
    mode = "classification",
    cost = tune(),
    degree = tune(),
    scale_factor = tune()
  ) %>%
  set_engine("kernlab")
  
# define workflows 
svm_poly_wflow <-
  workflow() %>%
  add_model(svm_poly_model) %>%
  add_recipe(recipe_para)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(svm_poly_model)

# change hyperparameter ranges 
svm_poly_params <- extract_parameter_set_dials(svm_poly_model)

# build tuning grid 
svm_poly_grid <- grid_latin_hypercube(svm_poly_params, size = 30)

# tune workflows/models ----
# set seed
set.seed(9403)

tic.clearlog() # clear log
tic("svm_poly") # start clock

tuned_svm_poly <-
  svm_poly_wflow %>%
  tune_grid(
    hotel_folds,
    grid = svm_poly_grid,
    control = control_stack_grid()
  )
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_poly <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

##----write out results (fitted/trained workflows & runtime info) ----

save(
  tuned_svm_poly,
  tictoc_svm_poly,
  file = here("results/fitted_tuned_models/ensemble/tuned_svm_poly.rda")
)