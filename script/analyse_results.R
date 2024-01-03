########## LOAD PACKAGES #####
library(dplyr)
library(ggplot2)

########## FORECAST #####
### 1) Load data
path_predictions <- "results/predictions"
ls_files <- list.files(path_predictions)
ls_files_full <- list.files(path_predictions, full.names = TRUE)
names(ls_files_full) <- gsub(ls_files, pattern = "_combined.csv", replacement = "")

df_all <- lapply(ls_files_full, read.csv) %>%
  bind_rows(.id = "model") %>%
  mutate(model = factor(model,
                        levels = c("GeneticSingleIs_GA",
                                   "GeneticSingleIs_RS",
                                   "SingleIs_GA",
                                   "enet_pred_RS",
                                   "xgb_pred_RS"),
                        labels = c("Reservoir FS (GA)",
                                   "Reservoir FS (RS)",
                                   "Reservoir no FS (RS)",
                                   "Elastic-net (RS)",
                                   "XGB (RS)")))

### 2) Performance dataframe
df_perf <- df_all %>%
  mutate(outcome = if_else(outcome < 10, 10, outcome),
         pred = if_else(pred < 10, 10, pred),
         hosp = if_else(hosp < 10, 10, hosp)) %>%
  group_by(outcomeDate, model) %>%
  summarise(outcome = unique(outcome),
            hosp = unique(hosp),
            pred = median(pred),
            .groups = "drop") %>%
  mutate(AE = abs(pred - outcome),
         RE = AE/outcome,
         baseline_AE = abs(hosp - outcome),
         AE_baseline = AE - baseline_AE,
         RE_baseline = AE/baseline_AE) %>%
  group_by(model) %>%
  summarise(AE = mean(AE, na.rm = T),
            AE_baseline = mean(AE_baseline, na.rm = T),
            RE = median(RE, na.rm = TRUE),
            RE_baseline = median(RE_baseline, na.rm = TRUE))

### 3) Graphical performance
plot_figure_performance <- df_all %>%
  filter(model %in% c("Reservoir FS (GA)",
                      "Elastic-net (RS)",
                      "XGB (RS)")) %>%
  group_by(model, outcomeDate) %>%
  group_by(outcomeDate, model) %>%
  summarise(outcome = unique(outcome),
            hosp = unique(hosp),
            pred = median(pred),
            .groups = "drop") %>%
  mutate(outcomeDate = as.Date(outcomeDate)) %>%
  ggplot(mapping = aes(x = outcomeDate, y = pred, color = model)) +
  geom_line() +
  geom_line(mapping = aes(y = outcome, color = " Observed")) +
  geom_line(mapping = aes(y = hosp, color = " Hosp t+14")) +
  scale_color_manual(values = c("darkgrey", "black", "#E9C46A", "#E76F51", "#2A9D8F")) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  labs(x = "Date",
       y = "Hospitalizations",
       color = "") +
  guides(color = guide_legend(ncol = 2))

### 4) Supplementary all graphical performance
plot_figure_performance_all <- df_all %>%
  group_by(model, outcomeDate) %>%
  group_by(outcomeDate, model) %>%
  summarise(outcome = unique(outcome),
            hosp = unique(hosp),
            pred = median(pred),
            .groups = "drop") %>%
  mutate(outcomeDate = as.Date(outcomeDate)) %>%
  ggplot(mapping = aes(x = outcomeDate, y = pred, color = "model")) +
  geom_line() +
  geom_line(mapping = aes(y = outcome, color = " Observed")) +
  geom_line(mapping = aes(y = hosp, color = " Hosp t+14")) +
  scale_color_manual(values = c("darkgrey", "black", "#E76F51")) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  labs(x = "Date",
       y = "Hospitalizations",
       color = "") +
  guides(color = guide_legend(ncol = 2)) +
  facet_wrap(model ~ .)

########## HYPER-PARAMETERS #####
path_hp <- "results/hyperparameter"
ls_files_full <- list.files(path_hp, full.names = TRUE, recursive = TRUE)
names(ls_files_full) <- gsub(ls_files_full, pattern = "results/hyperparameter/", replacement = "")

numeric_hp <- c(
  "ridge",
  "l1_ratio",
  "spectral_radius",
  "leaking_rate",
  "input_scaling",
  "n_estimators",
  "max_depth",
  "learning_rate",
  "subsample",
  "colsample_bytree"
)

df_all_hp <- lapply(ls_files_full,
                 function(x) read.csv(x) %>%
                     mutate_all(as.character)) %>%
  bind_rows(.id = "file_i") %>%
  mutate_at(
    .vars = c(numeric_hp, "time_seconds", "value"),
    .funs = as.numeric
  ) %>%
  tidyr::separate_wider_delim(cols = file_i,
                              delim = "/",
                              names = c("model", "date")) %>%
  mutate(model = factor(model,
                        levels = c("hp_GeneticSingleIs_GA",
                                   "hp_GeneticSingleIs_RS",
                                   "hp_SingleIs_GA",
                                   "hp_enet_pred_RS",
                                   "hp_xgb_pred_RS"),
                        labels = c("Reservoir FS (GA)",
                                   "Reservoir FS (RS)",
                                   "Reservoir no FS (RS)",
                                   "Elastic-net (RS)",
                                   "XGB (RS)")),
         date = gsub(pattern = ".csv", x = date, replacement = ""),
         date = as.Date(date),
         date = if_else(is.na(date), as.Date("2021-03-01"), date))

## save best model most important features
df_all_hp %>%
  filter(date == max(date)) %>%
  filter(model == "Reservoir FS (GA)") %>%
  slice_min(value, n = 1) %>%
  select(ends_with("_bin")) %>%
  tidyr::pivot_longer(cols = everything()) %>%
  filter(value == "y") %>%
  pull(name) %>%
  gsub(replacement = "", pattern = "_bin") %>%
  write("results/best_features.csv")

## get the best 40 by date
df_all_hp_best40 <- df_all_hp %>%
  group_by(model, date) %>%
  slice_min(value, n = 40)

##### show numeric hyperparameters
df_all_hp_best40_numeric <- df_all_hp_best40 %>%
  select(all_of(c("job_id", "model", "date", "value", numeric_hp))) %>%
  tidyr::pivot_longer(cols = numeric_hp,
                      values_to = "HP_value",
                      names_to = "HP_name") %>%
  na.omit()

ls_hpnum_plots <- df_all_hp_best40_numeric %>%
  group_by(model) %>%
  group_split() %>%
  lapply(FUN = function(df_i){
    model_title <- df_i$model %>% unique()
    df_i %>%
      mutate(date = as.factor(date)) %>%
      ggplot(mapping = aes(y = value, x = HP_value, color = date)) +
      geom_point() +
      facet_wrap(HP_name ~ ., scales = "free_x") +
      scale_color_viridis_d(direction = -1) +
      scale_x_log10() +
      theme_minimal() +
      labs(x = "Hyperparameter value",
           y = "MAE",
           color = "Update date",
           title = model_title) %>%
      return()
  })

##### show categorical hyperparameters
categorical_features <- grep(colnames(df_all_hp_best40), pattern = "_bin$", value = TRUE)
df_all_hp_best40_categorical <- df_all_hp_best40 %>%
  select(all_of(c("job_id", "model", "date", "value", categorical_features))) %>%
  tidyr::pivot_longer(cols = categorical_features,
                      values_to = "HP_value",
                      names_to = "HP_name") %>%
  na.omit() %>%
  group_by(model, date, HP_name) %>%
  summarise(freq_select = mean(HP_value == "y"),
            .groups = "drop")

ls_hpcat_plots <- df_all_hp_best40_categorical %>%
  group_by(model) %>%
  group_split() %>%
  lapply(FUN = function(df_i){
    
    model_title <- df_i$model %>% unique()
    
    vec_features_i <- df_i %>%
      filter(date == max(date)) %>%
      slice_max(freq_select, n = 10) %>%
      pull(HP_name)
    
    df_i %>%
      filter(HP_name %in% vec_features_i) %>%
      ggplot(mapping = aes(x = date, color = freq_select, y = HP_name)) +
      geom_point() +
      scale_color_viridis_c(direction = -1) +
      theme_minimal() +
      labs(x = "Update date",
           y = "",
           color = "Freq. select",
           title = model_title)
  })
