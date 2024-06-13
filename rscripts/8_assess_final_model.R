# ACCS Final Project ----
# Assess final model

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load testing and fitted data
load(here("data-splitting/hotel_test.rda")) 
load(here("results/final_fit.rda"))

# assessing models performance 
hotel_pred_prob <- final_fit %>% 
  predict(hotel_test, type = "prob")

hotel_pred_class <- final_fit %>%
  predict(hotel_test)

hotel_test_res <- hotel_test %>%
  bind_cols(hotel_pred_prob) %>%
  bind_cols(hotel_pred_class) %>% 
  select(is_canceled, .pred_class, .pred_0, .pred_1)

metrics <- metric_set(roc_auc, accuracy)
hotel_model_metrics <- metrics(hotel_test_res, truth = is_canceled, estimate = .pred_class, .pred_0)

roc_curve <- autoplot(roc_curve(hotel_test_res, is_canceled, .pred_0))

performance_table <-
  tibble(
    metrics = c("Accuracy", "ROC_AUC"),
    estimate = hotel_model_metrics %>% pull(.estimate)
  )

confusion_matrix <-
  conf_mat(hotel_test_res, truth = is_canceled, estimate = .pred_class)

heatmap <- autoplot(confusion_matrix, type = "heatmap") + 
  labs(title = "Confusion Matrix Heatmap") + theme_minimal() + 
  scale_fill_gradient(low = "gray", high = "lightblue") + 
  theme(legend.position = "none")

# save out results (plot, table)
write_rds(performance_table, here("results/performance_table.rds"))
write_rds(confusion_matrix, here("results/confusion_matrix.rds"))
ggsave(here("results/heatmap.png"))
ggsave(here("results/roc_curve.png"))
