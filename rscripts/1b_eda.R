# ACCS EDA ----
# In-Depth EDA after split
# BE AWARE: there are random processes in this script (seed set right before them)

# Load package(s)
library(tidymodels)
library(tidyverse)
library(here)
library(naniar)
library(corrplot)
library(ggcorrplot)

# handle common conflicts
tidymodels_prefer()

#load training data
load(here("data-splitting/hotel_train.rda"))

# set seed
set.seed(3463452)
hotel_eda <- hotel_train %>% slice_sample(prop = .8)
hotel_eda %>% skimr::skim_without_charts()

# univariate analysis ----

## continuous variables -----
cont_vars <- hotel_eda |> 
  select(where(is.numeric))

# function for distribution plots
dist_plots1 <- list()

for (var in colnames(cont_vars)) {
  # Create the distribution plot
  dist_plots1[[var]] <- ggplot(cont_vars, aes(x = !!sym(var))) + 
    geom_histogram(stat = "count") +
    labs(title = paste("Distribution of", var))
  
  # Save the plot as an image
  ggsave(paste0("images/univariate_plots/", var, ".png"), plot = dist_plots1[[var]], 
         height = 7, 
         width = 7, 
         units = "in")
}



## discrete variables -----
discrete_vars <- hotel_eda |> 
  select(where(is.factor))

# function for distribution plots
dist_plots2 <- list()

for(var in colnames(discrete_vars)) {
  if(is.factor(discrete_vars[[ var ]])) {
    dist_plots2[[ var ]] <- 
      ggplot(discrete_vars, aes( !! sym(var))) + 
      geom_bar() +
      labs(title = paste("Distribution of", var))
    
    print(dist_plots2[[var]])
  }
  ggsave(paste0("images/univariate_plots/", var, ".png"), plot = dist_plots2[[var]], 
         height = 7, 
         width = 7, 
         units = "in")
  
}


# bivariate analysis (with response variable) ----

## continuous variables -----
dist_plots3 <- list()

for (var in colnames(cont_vars)) {
  # Create the distribution plot
  dist_plots3[[var]] <- ggplot(cont_vars, aes(x = !!sym(var), y = is_canceled)) + 
    geom_boxplot() +
    labs(title = paste("Distribution of", var, "and is_canceled"))
  
  # Save the plot as an image
  ggsave(paste0("images/bivariate_plots/", var, ".png"), plot = dist_plots3[[var]], 
         height = 7, 
         width = 7, 
         units = "in")
}


## discrete variables ----
dist_plots4 <- list()

for(var in colnames(discrete_vars)) {
  if(is.factor(discrete_vars[[ var ]])) {
    dist_plots4[[ var ]] <- 
      ggplot(discrete_vars, aes( !! sym(var))) + 
      geom_bar(aes(fill = is_canceled)) +
      labs(title = paste("Distribution of", var, "and is_canceled"))
    
    print(dist_plots4[[var]])
  }
  ggsave(paste0("images/bivariate_plots/", var, ".png"), plot = dist_plots4[[var]], 
         height = 7, 
         width = 7, 
         units = "in")
  
}

## correlation ----

tmwr_cols <- colorRampPalette(c("#91CBD765", "#CA225E"))

cor_plot <- hotel_eda |>
  ungroup() |> 
  select(where(is.numeric)) |> 
  cor() |> 
  corrplot(col = tmwr_cols(200), 
           tl.col = "black", 
           method = "ellipse", 
           tl.cex = 0.5)
