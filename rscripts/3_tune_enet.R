# ACCS Final Project ----
# Define and fit Elastic Net model
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
enet_model <- 
  logistic_reg(
    mode = "classification",
    penalty = tune(),
    mixture = tune()
  ) %>%
  set_engine("glmnet")

# define workflows 
enet_wflow <-
  workflow() %>%
  add_model(enet_model) %>%
  add_recipe(recipe_para)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(enet_model)

# change hyperparameter ranges 
enet_params <- extract_parameter_set_dials(enet_model)

# build tuning grid 
enet_grid <- grid_regular(enet_params, levels = 5)

# tune workflows/models ----
# set seed
set.seed(4231)

tic("enet") # start clock

tuned_enet <-
  enet_wflow %>%
  tune_grid(
    hotel_folds,
    grid = enet_grid,
    control = control_grid(save_workflow = TRUE)
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_enet <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows)
save(
  tuned_enet,
  tictoc_enet,
  file = here("results/fitted_tuned_models/tuned_enet.rda")
)