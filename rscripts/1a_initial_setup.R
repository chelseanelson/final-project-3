# ACCS Initial Set Up ----
# Processing training, creating resamples, missingness & initial EDA
# BE AWARE: there are random processes in this script (seed set right before them)

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(naniar)

# handle common conflicts
tidymodels_prefer()


# load raw data
hotel <- read_csv(here("data/hotel_bookings.csv")) %>%
  mutate(across(where(is.character), as.factor),
         is_canceled = factor(is_canceled)) %>%
 # remove reservation date (already have arrival year, week, day), travel agent ID, company, and reservation status (tied to cancellation)
  select(-reservation_status_date, -agent, -company, -reservation_status)

# skim, check missingness
skimr::skim_without_charts(hotel)

# graph target var
canceled_og_distribution_plot <- hotel %>%
  ggplot(aes(is_canceled)) +
  geom_bar() + theme_minimal() + labs(
    title = "Distribution of Reservation Cancellations",
    x = "Cancelled?",
    y = "Count"
  ) +  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -1) + 
  scale_x_discrete(labels = c("Not Cancelled", "Cancelled"))

# downsampling
set.seed(9076)
hotel_small_0 <- hotel %>% 
  filter(!is.na(children), is_canceled == 0) %>% 
  slice_sample(n = 1500)

hotel_small_1 <- hotel %>% 
  filter(!is.na(children), is_canceled == 1) %>% 
  slice_sample(n = 1500)

hotel_small <- hotel_small_0 %>% bind_rows(hotel_small_1)

hotel_small %>% skimr::skim_without_charts()

# check downsampling
hotel_small %>%
  group_by(is_canceled) %>%
  summarise(count = n())

# target variable analysis after downsampling
canceled_balanced_distribution_plot <- hotel_small %>%
  ggplot(aes(is_canceled)) +
  geom_bar() + theme_minimal() + labs(
    title = "Distribution of Cancellation",
    x = "Canceled?",
    y = "Count"
  ) + 
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -1) + 
  scale_x_discrete(labels = c("Not Cancelled", "Cancelled"))
  
# initial split
hotel_split <- hotel_small %>%
  initial_split(prop = 0.75, strata = is_canceled)

hotel_train <- training(hotel_split)

hotel_test <- testing(hotel_split)

# resamples

hotel_folds <- hotel_train |>
  vfold_cv(v = 10, repeats = 5, strata = is_canceled)

# save splits and downsampled data
save(
  hotel_train,
  file = here("data-splitting/hotel_train.rda")
)

save(
  hotel_test,
  file = here("data-splitting/hotel_test.rda")
)

save(
  hotel_folds,
  file = here("data-splitting/hotel_folds.rda")
)

write_rds(
  hotel_small,
  file = here("data/hotel_small.rds")
)

# save plots 
ggsave(canceled_og_distribution_plot, file = here("figures/canceled_og_distribution_plot.png"))
ggsave(canceled_balanced_distribution_plot, file = here("figures/canceled_balanced_distribution_plot.png"))
