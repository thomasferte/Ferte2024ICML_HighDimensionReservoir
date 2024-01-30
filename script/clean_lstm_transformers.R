library(dplyr)

RCGA <- read.csv(file = "results/final_prediction/predictions/GeneticSingleIs_GA_combined.csv")
features <- read.csv(file = "results/best_features.csv")
nbFeatures <- features %>%
  mutate(last_used_observation = as.Date(last_used_observation)) %>%
  slice_min(last_used_observation) %>%
  nrow()

##### transformers #####

tranformers <- readODS::read_ods(path = "results/final_prediction/Resultat LSTM & Transformers/res_transformers.ods") %>%
  mutate(model = "transformer",
         trial = "transformer_1",
         nbFeatures = nbFeatures,
         pred = hosp + diff_predite,
         date = gsub(date, pattern = "'", replacement = "") %>% as.Date(.),
         outcomeDate = date) %>%
  select(trial, model, pred, nbFeatures, outcomeDate) %>%
  left_join(RCGA %>%
              slice_min(hp_date) %>%
              select(outcomeDate, outcome, hosp, mintraining, hp_date) %>%
              mutate(outcomeDate = as.Date(outcomeDate)) %>%
              distinct(),
            by = c("outcomeDate")) %>%
  filter(outcomeDate >= as.Date("2021-03-15")) %>%
  select(colnames(RCGA))

write.csv(tranformers, file = "results/final_prediction/predictions/transformers_combined.csv")

##### LSTM #####
ls_LSTM <- list.files(path = "results/final_prediction/Resultat LSTM & Transformers/LSTM_PCA/",
                      full.names = TRUE)

process_file <- function(file) {
  # Read the data from the file using read.csv
  df_res <- read.csv(file)
  
  # Extract trial and hp_date from the file path
  string_x <- unlist(strsplit(file, "/"))
  trial <- string_x[length(string_x) - 1]
  hp_date <- string_x[length(string_x) - 2]
  
  # Add trial and hp_date as new columns
  df_res$trial <- trial
  df_res$hp_date <- hp_date
  
  return(df_res)
}

# Use lapply to process each file and create a list of data frames
LSTM <- lapply(ls_LSTM, process_file) %>%
  bind_rows() %>%
  filter(outcomeDate >= as.Date("2021-03-15")) %>%
  select(colnames(RCGA))

write.csv(LSTM, file = "results/final_prediction/predictions/LSTM_combined.csv")
