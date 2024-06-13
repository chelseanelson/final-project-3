# Final Project----
# Assess trained ensemble model

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(here)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

#load stack
load(here("results/fitted_tuned_models/ensemble/hotel_st_blend.rda"))
load(here("results/fitted_tuned_models/ensemble/hotel_fit.rda"))


# Load testing data
load(here("data-splitting/hotel_test.rda"))
load(here("data-splitting/hotel_train.rda"))

#Load trained ensemble model info
load(here("results/fitted_tuned_models/ensemble/hotel_fit.rda"))
load(here("results/fitted_tuned_models/ensemble/tuned_nnet.rda"))
load(here("results/fitted_tuned_models/ensemble/tuned_rf.rda"))
load(here("results/fitted_tuned_models/ensemble/tuned_svm_poly.rda"))


##----Assess-ensemble-model---- 
svm_params <- hotel_fit |> 
  collect_parameters(candidates = "tuned_svm_poly") |> 
  filter(coef != 0)

rf_params <- hotel_fit |> 
  collect_parameters(candidates = "tuned_rf") |> 
  filter(coef != 0)

nnet_params <- hotel_fit |> 
  collect_parameters(candidates = "tuned_nnet") |> 
  filter(coef != 0)

# binding all models 
all_params <- bind_rows(rf_params, nnet_params, svm_params) |> 
  rename(
    weights = coef
  )


##----predictions---- 
metrics <- metric_set(roc_auc, accuracy)

hotel_pred_prob <- hotel_fit %>% 
  predict(hotel_test, type = "prob")

hotel_pred_class <- hotel_fit %>%
  predict(hotel_test)

hotel_test_res <- hotel_test %>%
  bind_cols(hotel_pred_prob) %>%
  bind_cols(hotel_pred_class) %>% 
  select(is_canceled, .pred_class, .pred_0, .pred_1)

ensemble_performance_metrics <- metrics(hotel_test_res, truth = is_canceled, estimate = .pred_class,.pred_0) %>%
  select(-.estimator) %>% rename(metric = .metric, estimate = .estimate)

##----save out results----
save(svm_params, file = here("results/fitted_tuned_models/ensemble/svm_params.rds"))
save(rf_params, file = here("results/fitted_tuned_models/ensemble/rf_params.rds"))
save(nnet_params, file = here("results/fitted_tuned_models/ensemble/nnet_params.rds"))
write_rds(ensemble_performance_metrics, file = here("results/fitted_tuned_models/ensemble/ensemble_performance_metrics.rds"))
