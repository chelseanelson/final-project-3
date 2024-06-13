# Classification Problem ----
# Variable selection using lasso regression

# load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)

# handle common conflicts
tidymodels_prefer()

# parallel processing ----
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)

# load trainind data 
load(here("data-splitting/hotel_train.rda"))

# create resamples/folds ----

# only 1 repeat
set.seed(1097)
lasso_folds <- 
  hotel_train |> 
  vfold_cv(v = 5, repeats = 1, strata = is_canceled)


### NO ONE_HOT for linear models
# basic recipe ----
recipe_basic <- recipe(is_canceled ~., data = hotel_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>% 
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

# check recipe
recipe_basic |> 
  prep() |> 
  bake(new_data = NULL)

# model specifications ----
lasso_spec <- logistic_reg(
  mixture = 1, 
  penalty = tune()
) |> 
  set_mode("classification") |> 
  set_engine("glmnet")

# define workflows ----
lasso_wflow <- 
  workflow() |> 
  add_model(lasso_spec) |> 
  add_recipe(recipe_basic)

# hyperparameter tuning values ----
hardhat::extract_parameter_set_dials(lasso_spec)

# build tuning grid
lasso_params <- hardhat::extract_parameter_set_dials(lasso_spec) |> 
  update(penalty = penalty(c(-3,0)))

lasso_grid <- grid_regular(lasso_params, levels = 5)

# fit workflow/model ----
# extract best model (optimal tuning parameters)
lasso_tuned <- 
  lasso_wflow |> 
  tune_grid(
    resamples = lasso_folds, 
    grid = lasso_grid, 
    metrics = metric_set(roc_auc), 
    control = control_grid(save_workflow = TRUE)
  )

lasso_plot <- autoplot(lasso_tuned)
# fit best model/results
optimal_wflow <- extract_workflow(lasso_tuned) |> 
  finalize_workflow(select_best(lasso_tuned, metric = "roc_auc"))

# fit best model/results---- 
var_select_fit_lasso <- fit(optimal_wflow, hotel_train)

# write out variable selection results ----
save(var_select_fit_lasso, file = here("rscripts/var-selection/results/var_select_fit_lasso.rda"))