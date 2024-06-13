# ACCS Final Project ----
# Define and fit Random Forest model, with lasso recipe
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
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
load(here("recipes/recipe_lasso.rda"))

# model specifications ----
rf_model <- 
  rand_forest(
    mode = "classification",
    min_n = 2,
    mtry = tune(),
    trees = 1000
  ) %>%
  set_engine("ranger")

# define workflows 
rf_wflow <-
  workflow() %>%
  add_model(rf_model) %>%
  add_recipe(recipe_lasso)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(rf_model) 

# change hyperparameter ranges 
rf_params <- extract_parameter_set_dials(rf_model) |> 
  update(
    mtry = mtry(c(0,40))
  )

# build tuning grid 
rf_grid <- grid_latin_hypercube(rf_params, size = 15)

# tune workflows/models ----
tic.clearlog() # clear log
tic("rf-improved") # start clock

# set seed
set.seed(30091)
improve_rf1 <-
  rf_wflow %>%
  tune_grid(
    hotel_folds,
    grid = rf_grid,
    control = control_grid(save_workflow = TRUE)
  )

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_rf_improved <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)



# write out results (fitted/trained workflows)
save(improve_rf1, tictoc_rf_improved, file = here("results/fitted_tuned_models/tuned_rf_improve.rda"))
