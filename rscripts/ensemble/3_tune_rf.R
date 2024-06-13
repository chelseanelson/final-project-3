# ACCS Final Project ----
# Define and fit Random Forest model
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)
library(tictoc)
library(doMC)

# handle common conflicts 
tidymodels_prefer()

# set up parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load folded data 
load(here("data-splitting/hotel_folds.rda"))

# load pre-processing/feature engineering recipe 
load(here("recipes/2_recipe_nonpara.rda"))

# model specifications ----
rf_model <- 
  rand_forest(
    mode = "classification",
    min_n = tune(),
    mtry = tune(),
    trees = 1000
  ) %>%
  set_engine("ranger")

# define workflows 
rf_wflow <-
  workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe_nonpara)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(rf_model) 

# change hyperparameter ranges 
rf_params <- extract_parameter_set_dials(rf_model) |> 
  update(
    mtry = mtry(c(20,40))
  )

# build tuning grid 
rf_grid <- grid_regular(rf_params, levels)

# tune workflows/models ----
tic.clearlog() # clear log
tic("RF: Kitchen Sink") # start clock

# set seed
set.seed(31011)
tuned_rf <-
  rf_wflow %>%
  tune_grid(
    hotel_folds,
    grid = rf_grid,
    control = control_stack_grid()
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_rf_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)



# write out results (fitted/trained workflows)
save(tictoc_rf_ks, 
     tuned_rf, file = here("results/fitted_tuned_models/ensemble/tuned_rf.rda"))

