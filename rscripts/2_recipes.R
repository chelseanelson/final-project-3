# ACCS Recipes ----
# Creating recipes 

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data 
load(here("data-splitting/hotel_train.rda"))

# Featured Engineered Recipes 1 ----

## recipe 1 (naive bayes) ----
recipe_naivebayes <- recipe(is_canceled ~., data = hotel_train) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_other(country, threshold = 0.05) %>% 
  step_nzv(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors())

# check recipe
recipe_naivebayes %>% 
  prep() %>% 
  bake(new_data = NULL) %>%
  glimpse()

## recipe 2 (parametric) ----

recipe_para <- recipe(is_canceled ~., data = hotel_train) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors())

# check recipe
recipe_para %>% 
  prep() %>% 
  bake(new_data = NULL) %>% 
  glimpse()

## recipe 3 (nonparametric) ----

recipe_nonpara <- recipe(is_canceled ~., data = hotel_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>% 
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

#check recipe
recipe_nonpara %>% 
  prep() %>%
  bake(new_data = NULL) %>%
  glimpse()

# Feature Engineered Recipes 2 ----

## recipe 2 (parametric) ----

recipe_para_2 <- recipe(is_canceled ~., data = hotel_train) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_other(all_nominal_predictors(), threshold = 0.05) %>%
  step_corr(all_numeric_predictors()) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) %>% 
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune())

# check recipe
# recipe_para_2 %>% 
#   prep() %>% 
#   bake(new_data = NULL) %>% 
#   glimpse()

## recipe 3 (nonparametric) ----

recipe_nonpara_2 <- recipe(is_canceled ~., data = hotel_train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>% 
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), num_comp = tune())

# check recipe
# recipe_nonpara_2 %>% 
#   prep() %>% 
#   bake(new_data = NULL) %>% 
#   glimpse()


# ###############################################################################
# # Recipe with variables selected by lasso regression
 ###############################################################################

# set up for this recipe can be found in (`rscripts/var-selection/setting-recipes.R`)
# not functionable here, only for show

recipe_lasso <- recipe(is_canceled ~ ., data = hotel_train) |>
  step_rm(any_of( !!var_remove )) |>
  step_novel(all_nominal_predictors()) %>%
  step_other(all_nominal_predictors(), threshold = 0.05) %>% 
  step_corr(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_nzv(all_predictors()) %>%
  step_normalize(all_predictors())

recipe_lasso |>
  prep() |>
  bake(new_data = NULL) |>
  glimpse()

# save recipes ----
save(recipe_naivebayes, file = here("recipes/2_recipe_naivebayes.rda"))

save(recipe_para, file = here("recipes/2_recipe_para.rda"))

save(recipe_nonpara, 
     file = here("recipes/2_recipe_nonpara.rda"))


save(recipe_para_2, file = here("recipes/2_recipe_para_2.rda"))

save(recipe_nonpara_2, 
     file = here("recipes/2_recipe_nonpara_2.rda"))

# save lasso recipe
save(recipe_lasso, file = here("recipes/2_recipe_lasso.rda"))
  
