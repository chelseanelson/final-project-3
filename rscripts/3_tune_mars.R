# ACCS Final Project ----
# Define and fit MARS model
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
mars_model <-
  mars(
    mode = "classification",
    num_terms = tune(),
    prod_degree = tune()
  ) %>%
  set_engine("earth")

# define workflows
mars_wflow <-
  workflow() %>%
  add_model(mars_model) %>%
  add_recipe(recipe_para)

# hyperparameter tuning values ----

# check ranges for hyperparameters
hardhat::extract_parameter_set_dials(mars_model)

# change hyperparameter ranges
mars_params <- extract_parameter_set_dials(mars_model)

# build tuning grid
mars_grid <- grid_latin_hypercube(mars_params, size = 30)

# tune workflows/models ----
# set seed
set.seed(2423)

tic("MARS")

tuned_mars <-
  mars_wflow %>%
  tune_grid(
    hotel_folds,
    grid = mars_grid,
    control = control_grid(save_workflow = TRUE)
  )

toc(log = TRUE)
time_log <- tic.log(format = FALSE)

tictoc_mars <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows)
save(tuned_mars, tictoc_mars,
     file = here("results/fitted_tuned_models/tuned_mars.rda"))
