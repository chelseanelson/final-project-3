# ACCS Final Project ----
# Define and fit Neural Network model
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
registerDoMC(cores = num_cores)

# load folded data
load(here("data-splitting/hotel_folds.rda"))

# load pre-processing/feature engineering recipe
load(here("recipes/2_recipe_nonpara.rda"))

# model specifications ----
nnet_model <-
  mlp(
    mode = "classification",
    hidden_units = tune(),
    penalty = tune()
  ) %>%
  set_engine("nnet")

# define workflows
nnet_wflow <-
  workflow() %>%
  add_model(nnet_model) %>%
  add_recipe(recipe_nonpara)

# hyperparameter tuning values ----

# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(nnet_model)

# change hyperparameter ranges
nnet_params <- extract_parameter_set_dials(nnet_model)

# build tuning grid
nnet_grid <- grid_latin_hypercube(nnet_params, size = 30)

# tune workflows/models ----
# set seed
set.seed(423123)

tic("NNET")

tuned_nnet <-
  nnet_wflow %>%
  tune_grid(
    hotel_folds,
    grid = nnet_grid,
    control = control_stack_grid()
  )

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

tictoc_nnet <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows)
save(tuned_nnet, tictoc_nnet,
     file = here("results/fitted_tuned_models/ensemble/tuned_nnet.rda"))
