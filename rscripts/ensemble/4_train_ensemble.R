# ACCS Final Project ----
# Define and fit Ensemble model
# BE AWARE: there is a random process in this script (seed set right before it)

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)
library(doMC)

# Handle common conflicts
tidymodels_prefer()

# set up parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# Load candidate model info ----
load(here("results/fitted_tuned_models/ensemble/tuned_nnet.rda"))
load(here("results/fitted_tuned_models/ensemble/tuned_rf.rda"))
load(here("results/fitted_tuned_models/ensemble/tuned_svm_poly.rda"))

# Create data stack ----
hotel_data_st <- 
  stacks() %>%
  add_candidates(tuned_svm_poly) %>%
  add_candidates(tuned_nnet) %>%
  add_candidates(tuned_rf)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions (tuning step, set seed)
set.seed(19234)
hotel_st_blend <-
  hotel_data_st %>%
  blend_predictions(penalty = blend_penalty)

# Save blended model stack for reproducibility & easy reference (for report)
save(hotel_st_blend, file = here("results/fitted_tuned_models/ensemble/hotel_st_blend.rda"))


##---- fit to training set ----
hotel_fit <-
  hotel_st_blend %>%
  fit_members()

##----weights----
autoplot(hotel_fit, type = "weights")
##----members----
autoplot(hotel_fit, type = "members")

# Save trained ensemble model for reproducibility & easy reference (for report)
save(hotel_fit, file = here("results/fitted_tuned_models/ensemble/hotel_fit.rda"))
