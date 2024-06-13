# ACCS Final Project
# Analysis of improved tuned and trained models (comparison)
# Main Assessment Metric : ROC_AUC

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load in tuned and fitted models
load(here("results/fitted_tuned_models/tuned_nnet_improve.rda"))
load(here("results/fitted_tuned_models/tuned_svm_poly_improve.rda"))
load(here("results/fitted_tuned_models/tuned_rf_improve.rda"))

runtime <- bind_rows(tictoc_nnet_improve,
                     tictoc_svm_poly,
                     tictoc_rf_improved
) %>%
  select(model, runtime) %>% mutate(model = case_when(
    model == "nnet_improve" ~ "nnet",
    model == "improve_svm_poly" ~ "svm_poly",
    model == "rf-improved" ~ "rf",
    TRUE ~ model
  ))

# select best of each
get_best <- function(tuned_model, name) {
  show_best(tuned_model, metric = "roc_auc") %>%
    mutate(model = name) %>%
    slice_head(n = 1) %>%
    select(model, mean, std_err)
}

nnet <- get_best(tuned_nnet_improve, "nnet")

svm_poly <- get_best(tuned_svm_poly, "svm_poly")

rf <- get_best(improve_rf1, "rf")


# combine best and runtime
improved_results <- bind_rows(nnet,
                             svm_poly,
                             rf
) %>%
  left_join(runtime) %>% arrange(desc(mean))





# comparing sub-models ----

## Neural Network
show_best(tuned_nnet_improve, metric = "roc_auc")
autoplot(tuned_nnet_improve, metric = "roc_auc")

## Random Forest
show_best(improve_rf1, metric = "roc_auc")
autoplot(improve_rf1, metric = "roc_auc")

## SVM Polynomial
show_best(tuned_svm_poly, metric = "roc_auc")
# autoplot(tuned_svm_poly, metric = "roc_auc")


# Best Model: Improved Random Forest 

### write out results (plots, tables)
save(improved_results, file = here("results/improved_results.rda"))
