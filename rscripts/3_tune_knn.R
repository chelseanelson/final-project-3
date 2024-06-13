# ACCS Final Project ----
# Define and fit k-nearest neighbors model
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
load(here("recipes/2_recipe_nonpara.rda"))

# model specifications ----
knn_model <- 
  nearest_neighbor(
    mode = "classification",
    neighbors = tune()
  ) %>%
  set_engine("kknn")

# define workflows 
knn_wflow <-
  workflow() %>%
  add_model(knn_model) %>%
  add_recipe(recipe_nonpara)

# hyperparameter tuning values ----

# check ranges for hyperparameters 
hardhat::extract_parameter_set_dials(knn_model)

# change hyperparameter ranges 
knn_params <- extract_parameter_set_dials(knn_model)

# build tuning grid 
knn_grid <- grid_regular(knn_params, levels)

# tune workflows/models ----
# set seed
set.seed(4231011)
tic.clearlog() # clear log
tic("knn") # start clock

tuned_knn <-
  knn_wflow %>%
  tune_grid(
    hotel_folds,
    grid = knn_grid,
    control = control_grid(save_workflow = TRUE)
  )
toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_knn <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# autoplot 
knn_plot <- autoplot(tuned_knn, metric = "roc_auc")

# write out results (fitted/trained workflows)
save(tuned_knn,tictoc_knn, file = here("results/fitted_tuned_models/tuned_knn.rda"))
ggsave(knn_plot, file = here("images/plots/knn_plot.png"))