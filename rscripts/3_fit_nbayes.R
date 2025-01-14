# ACCS Final Project ----
# Define and fit naive bayes model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(discrim)
library(tictoc)

# handle common conflicts 
tidymodels_prefer()

# set up parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load folded data 
load(here("data-splitting/hotel_folds.rda"))

# load pre-procesing/feature engineering recipe 
load(here("recipes/2_recipe_naivebayes.rda"))

# model specifications ----
nbayes_model <-
  naive_Bayes(mode = "classification") %>% 
  set_engine("klaR")

# define workflows ---
nbayes_wflow <-
  workflow() %>% 
  add_model(nbayes_model) %>%
  add_recipe(recipe_naivebayes)

# fit workflows/models
tic("nbayes") # start clock

keep_wflow <- control_resamples(save_workflow = TRUE)

fit_nbayes <-
  fit_resamples(
    nbayes_wflow,
    resamples = hotel_folds,
    control = keep_wflow
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_nbayes <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows) ----
save(
  fit_nbayes,
  tictoc_nbayes,
  file = here("results/fitted_tuned_models/fit_nbayes.rda")
)