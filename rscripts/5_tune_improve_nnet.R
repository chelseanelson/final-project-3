# ACCS Final Project ----
# Define and fit Neural Network model (improved)
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
load(here("recipes/2_recipe_nonpara_2.rda"))

# model specifications ----
nnet_improve_model <-
  mlp(
    mode = "classification",
    hidden_units = tune(),
    penalty = tune()
  ) %>%
  set_engine("nnet")

# define workflows
nnet_improve_wflow <-
  workflow() %>%
  add_model(nnet_improve_model) %>%
  add_recipe(recipe_nonpara_2)

# hyperparameter tuning values ----

# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(nnet_improve_model)

# change hyperparameter ranges
nnet_improve_params <- extract_parameter_set_dials(recipe_nonpara_2) |>
  bind_rows(extract_parameter_set_dials(nnet_improve_model))

# build tuning grid
nnet_improve_grid <- grid_latin_hypercube(nnet_improve_params, size = 30)

# tune workflows/models ----
# set seed
set.seed(423123)

tic("nnet_improve")

tuned_nnet_improve <-
  nnet_improve_wflow %>%
  tune_grid(
    hotel_folds,
    grid = nnet_improve_grid,
    control = control_grid(save_workflow = TRUE)
  )

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

tictoc_nnet_improve <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows)
save(tuned_nnet_improve, tictoc_nnet_improve,
     file = here("results/fitted_tuned_models/tuned_nnet_improve.rda"))
