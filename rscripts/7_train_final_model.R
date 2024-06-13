# ACCS Final Project ----
# Train final model
# Best Model: Improved Random Forest 
# BE AWARE: there is a random process in this script (seed set right before it)

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# best model: improved Random Forest 

# load tuned and training data 
load(here("results/fitted_tuned_models/tuned_rf_improve.rda"))
load(here("data-splitting/hotel_train.rda"))

# finalize workflow ----
final_wflow <- improve_rf1 %>%
  extract_workflow(improve_rf1) %>%
  finalize_workflow(select_best(improve_rf1, metric = "roc_auc"))


# train final model ----
set.seed(390)
final_fit <- fit(final_wflow, hotel_train)

# write out fitted data
save(final_fit, file = here("results/final_fit.rda"))
