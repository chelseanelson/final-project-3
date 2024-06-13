# Final Project Repository

Welcome to our final project repository! This project is dedicated to addressing the prediction problem of hotel cancellations using a comprehensive dataset of real hotel information. Thus we want to see if 
a hotel reservation will be canceled given characteristics of the booking and hotel stay. Through rigorous analysis and model development, we have identified key insights, tackled class imbalance, and culminated in selecting our improved random forest model as our top performer, achieving an impressive roc_auc rate of __%. Thus delving into the intricacies of predictive modeling in the hotel industry, striving to contribute valuable insights to the field. 

## Project Structure

The Final Project to predict cancelled hotel reservations includes the following: 

### Folders

`data/`: contains datasets used in model building 

`data-splitting/`: contains all the data splits (testing, training, and V-folded data)

`images/`: stores visualizations and figures generated during the project

`memos/`: contains all progress memos

`recipes/`: contains the recipes created to be used with model building 

`results/`: contains results associated with the tuned and final models 

`rscripts/`: contains all of the rscripts 

### Usage 

If you want to explore or run the code, we would recommend starting with `rscripts/1a_initial_setup.R` to understand the distribution and layout of the original data and then running in order after that in order to see the development and creation of the different types of models. We would finish off by looking at `rscripts/4_model_analysis.R` through `rscripts/8_assess_final_model.R` to see how each model performs, looking at which performances the best with ROC_AUC. Otherwise, all of the associated model data, fitted data, tables of the performance analyses, and the recipes in which we created can be found in their respective rscripts.

Additionally, however, if you would just like a quick overview of our project, we would recommend reading through both `memos/accs_progress_memo_1.html` and `memos/accs_progress_memo_2.html` as well as `accs_final_report.html` for a complete overview of my project, from start to finish.

If you would like an even a quicker overview of the main insights generated from our project, we would recommend reading through `accs_executive_summary.html` as that is a short executive summary of all the work and insights that we gained throughout our work.

## Acknowledgements
We call any external libraries used at the top of each rscript file if needed throughout looking through our progress.

All references to external sources and information used have been highlighted and can be found in the References section of `accs_final_report.html`