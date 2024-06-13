# ACCS Final Project
# Analysis of tuned and trained models (comparison)
# Main Assessment Metric : ROC_AUC

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(stacks)

# handle common conflicts
tidymodels_prefer()

##----tbl-initial-results----
# load in tuned and fitted models
load(here("results/fitted_tuned_models/fit_nbayes.rda"))
load(here("results/fitted_tuned_models/tuned_enet.rda"))
load(here("results/fitted_tuned_models/tuned_knn.rda"))
load(here("results/fitted_tuned_models/tuned_mars.rda"))
load(here("results/fitted_tuned_models/tuned_nnet.rda"))
load(here("results/fitted_tuned_models/tuned_rf.rda"))
load(here("results/fitted_tuned_models/tuned_svm_poly.rda"))
load(here("results/fitted_tuned_models/tuned_svm_radial.rda"))
load(here("results/fitted_tuned_models/ensemble/hotel_st_blend.rda"))

# bind tictoc results
runtime <- bind_rows(tictoc_nbayes,
                     tictoc_enet,
                     tictoc_mars,
                     tictoc_nnet,
                     tictoc_svm_radial,
                     tictoc_svm_poly,
                     tictoc_knn,
                     tictoc_rf
                    ) %>%
  select(model, runtime)

# select best of each
get_best <- function(tuned_model, name) {
  show_best(tuned_model, metric = "roc_auc") %>%
    mutate(model = name) %>%
    slice_head(n = 1) %>%
    select(model, mean, std_err)
}

nbayes <- get_best(fit_nbayes, "nbayes")

enet <- get_best(tuned_enet, "enet")

mars <- get_best(tuned_mars, "MARS")

nnet <- get_best(tuned_nnet, "NNET")

svm_radial <- get_best(tuned_svm_radial, "svm_radial")

svm_poly <- get_best(tuned_svm_poly, "svm_poly")

knn <- get_best(tuned_knn, "knn")

rf <- get_best(tuned_rf, "rf")


# combine best and runtime
initial_results <- bind_rows(nbayes,
          enet,
          mars,
          nnet,
          svm_radial,
          svm_poly,
          knn,
          rf
) %>%
  left_join(runtime) %>% arrange(desc(mean)) |> 
  mutate(model = case_when(
    model == "NNET" ~ "nnet",
    TRUE ~ model
  ))

models_to_improve_results <- bind_rows(
                             nnet,
                             svm_poly,
                             rf
) %>%
  left_join(runtime) %>% arrange(desc(mean)) |> 
  mutate(model = case_when(
    model == "NNET" ~ "nnet",
    TRUE ~ model
  ))

# Autoplot Top Three Models
initial_results %>%
  knitr::kable(digits = 4)

##---- Autoplot Top Three Models ----
autoplot(tuned_rf, metric = "roc_auc")
autoplot(tuned_svm_poly, metric = "roc_auc")
autoplot(tuned_nnet, metric = "roc_auc")

##----comparing-sub-models ----
## Naive Bayes
nbayes_best <- show_best(fit_nbayes, metric = "roc_auc") %>% 
  knitr::kable()

## Elastic Net
# enet_plot <- tuned_enet %>% autoplot(metric = "roc_auc")
# enet_best <- show_best(tuned_enet, metric = "roc_auc") %>% 
#   mutate(penalty = as.character(round(penalty, 10))) %>% 
#   knitr::kable(digits = 4)

## Neutral Networks
nnet_plot <- tuned_nnet %>% autoplot(metric = "roc_auc")
nnet_best <- show_best(tuned_nnet, metric = "roc_auc") %>% 
  knitr::kable(digits = 4)

## K-Nearest Neighbors
knn_plot <- tuned_knn %>% autoplot(metric = "roc_auc")
knn_best <- show_best(tuned_knn, metric = "roc_auc")

## MARS
mars_plot <- tuned_mars %>% autoplot(metric = "roc_auc")
mars_best <- show_best(tuned_mars, metric = "roc_auc") %>% 
  knitr::kable(digits = 4)

## Random Forests
rf_plot <- tuned_rf %>% autoplot(metric = "roc_auc")
rf_best <- show_best(tuned_rf, metric = "roc_auc") %>% 
  knitr::kable(digits = 4)

## SVM Polynomial
svm_poly_plot <- tuned_svm_poly %>% autoplot(metric = "roc_auc")
svm_poly_best <- show_best(tuned_svm_poly, metric = "roc_auc") %>% 
  knitr::kable(digits = 4)

## SVM Radial
svm_radial_plot <- tuned_svm_radial %>% autoplot(metric = "roc_auc")
svm_radial_best <- show_best(tuned_svm_radial, metric = "roc_auc") %>% 
  knitr::kable(digits = 4)

## Ensemble
# Explore the blended model stack
ensemble_plot <- autoplot(hotel_st_blend)
ensemble_weights_plot <- autoplot(hotel_st_blend, type = "weights")

# write out results (plots, tables)
save(initial_results, file = here("results/initial_results.rda"))
save(rf_best, file = here("results/rf_best.rda"))
ggsave(rf_plot, file = here("images/autoplots/rf_plot.png"))
save(svm_poly_best, file = here("results/svm_poly_best.rda"))
ggsave(svm_poly_plot, file = here("images/autoplots/svm_poly_plot.png"))
save(nnet_best, file = here("results/nnet_best.rda"))
ggsave(nnet_plot, file = here("images/autoplots/nnet_plot.png"))
save(models_to_improve_results, file = here("results/models_to_improve_results.rda"))
