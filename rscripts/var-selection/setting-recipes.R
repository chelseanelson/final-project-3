# Final Project---- 
# setup pre-processing recipes 

# load packages ---- 
library(tidyverse)
library(tidymodels)
library(here)
library(recipes)

# handle common conflicts
tidymodels_prefer()

# load training data and var selection results
load(here("data-splitting/hotel_train.rda"))
load(here("rscripts/var-selection/results/var_select_fit_lasso.rda"))

var_select_lasso <- var_select_fit_lasso |> tidy()

## getting numerical variables
numeric_vars <- hotel_train |> 
  select(where(is.numeric)) |> 
  colnames()

# getting factor vars
factor_vars <- hotel_train |> 
  select(where(is.factor)) |> 
  colnames()

# important variables
imp_vars <- var_select_lasso |> 
  filter(estimate != 0) |> 
  pull(term)


## getting numeric ----
imp_numeric <- imp_vars[imp_vars %in% numeric_vars]


## getting factor is tricky bc of the renaming ----
num_true <- map( 
  factor_vars, 
  # if important vars start with the factor = TRUE
  ~ startsWith(imp_vars, prefix = .x) |> 
    # sum tells us how many levels were important 
    sum()
)

## assign raw names from dataset -----
names(num_true) <- factor_vars

## if at least one factor level was important, lets keep it----
imp_factor <- enframe(unlist(num_true)) |> 
  filter(value != 0) |> # or greater than x
  pull(name)

var_keep <- c(imp_numeric, imp_factor)

# finds columns that are not in var_keep, is_canceled 
var_remove <- setdiff( 
  names(hotel_train), 
  c(var_keep, "is_canceled"))


# ###############################################################################
# # Recipe with variables selected by lasso regression
# ###############################################################################
# 
recipe_lasso <- recipe(is_canceled ~ ., data = hotel_train) |>
  step_rm(any_of( !!var_remove )) |>
  step_date(all_date_predictors(), keep_original_cols = FALSE) |>
  step_impute_mean(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_other(all_nominal_predictors(), threshold = .05) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_zv(all_predictors()) |>
  step_normalize(all_predictors())

recipe_lasso |>
  prep() |>
  bake(new_data = NULL) |>
  glimpse()

# save lasso recipe
save(recipe_lasso, file = here("recipes/recipe_lasso.rda"))
