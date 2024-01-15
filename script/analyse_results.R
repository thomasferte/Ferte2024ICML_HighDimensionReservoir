########## LOAD PACKAGES #####
library(dplyr)
library(parallel)
library(ggplot2)
source(file = "script/FctCleanFeaturesName.R")
set.seed(1)

path_results <- "results/monthly_from_scratch/"
results_save <- "results/monthly_from_scratch/"

########## NB FEATURES #####
read.csv("data_obfuscated/20220117.csv") %>%
  select(-outcome, -outcomeDate, -START_DATE) %>%
  dim()
########## Understand gaps #####
df_data <- read.csv("data_obfuscated/20220117.csv")

df_data %>%
  select(P_TOUS_AGES, TESTED_TOUS_AGES, IPTCC.mean, outcome, outcomeDate, hosp) %>%
  mutate(outcomeDate = as.Date(outcomeDate)) %>%
  tidyr::pivot_longer(cols = -outcomeDate) %>%
  ggplot(mapping = aes(x = outcomeDate, y = value)) +
  geom_line() +
  facet_grid(name ~ ., scales = "free_y") +
  scale_x_date(date_breaks = "months")

########## FORECAST #####
### 1) Load data
path_predictions <- paste0(path_results, "predictions/")
ls_files <- list.files(path_predictions)
ls_files_full <- list.files(path_predictions, full.names = TRUE)
names(ls_files_full) <- gsub(ls_files, pattern = "_combined.csv", replacement = "")

df_all_temp <- lapply(ls_files_full, read.csv) %>%
  bind_rows(.id = "model") %>%
  mutate(model = factor(model,
                        levels = c("GeneticSingleIs_GA",
                                   "GeneticSingleIs_RS",
                                   "SingleIs_GA",
                                   "enet_pred_RS",
                                   "xgb_pred_RS"),
                        labels = c("Reservoir FS (GA)",
                                   "Reservoir FS (RS)",
                                   "Reservoir no FS (GA)",
                                   "Elastic-net (RS)",
                                   "XGB (RS)"))) %>%
  group_by(outcomeDate, model)

df_all <- df_all_temp %>% slice_min(hp_date) %>% mutate(update = "No") %>%
  bind_rows(df_all_temp %>% slice_max(hp_date) %>% mutate(update = "Yes")) %>%
  ungroup()

### 2) Performance dataframe
df_perf <- df_all %>%
  mutate(outcome = if_else(outcome < 10, 10, outcome),
         pred = if_else(pred < 10, 10, pred),
         hosp = if_else(hosp < 10, 10, hosp)) %>%
  group_by(outcomeDate, model, update) %>%
  summarise(outcome = unique(outcome),
            hosp = unique(hosp),
            pred = median(pred),
            .groups = "drop") %>%
  mutate(AE = abs(pred - outcome),
         RE = AE/outcome,
         baseline_AE = abs(hosp - outcome),
         AE_baseline = AE - baseline_AE,
         RE_baseline = AE/baseline_AE) %>%
  group_by(model, update) %>%
  summarise(AE = mean(AE, na.rm = T),
            AE_baseline = mean(AE_baseline, na.rm = T),
            RE = median(RE, na.rm = TRUE),
            RE_baseline = median(RE_baseline, na.rm = TRUE))

# df_perf %>% knitr::kable(format = "latex", booktabs = TRUE, digits = 2)

### ESN number of model needed
nboot <- 250
vec_nb_esn <- c(1:10, 20, 40)

dfreservoir <- df_all %>%
  filter(model == "Reservoir FS (GA)", update == "No") %>%
  mutate(outcome = if_else(outcome < 10, 10, outcome),
         pred = if_else(pred < 10, 10, pred),
         hosp = if_else(hosp < 10, 10, hosp))

dfRepeatReservoir <- mclapply(X = seq_len(nboot),
                              mc.cores = parallel::detectCores()-2,
                              function(boot_i){
                                lapply(X = vec_nb_esn,
                                       function(nb_esn){
                                         MAE <- dfreservoir %>%
                                           group_by(outcomeDate) %>%
                                           sample_n(nb_esn, replace = TRUE) %>%
                                           summarise(outcome = unique(outcome),
                                                     hosp = unique(hosp),
                                                     pred = median(pred),
                                                     .groups = "drop") %>%
                                           mutate(AE = abs(pred - outcome)) %>%
                                           pull(AE) %>%
                                           mean(na.rm = T)
                                         
                                         data.frame(nb_esn = nb_esn,
                                                    MAE = MAE)
                                       }) %>%
                                  bind_rows()
                              }) %>%
  bind_rows(.id = "boot_i")

plot_repeated_reservoir <- dfRepeatReservoir %>%
  mutate(MAE = if_else(MAE > 60, 60, MAE)) %>%
  group_by(nb_esn) %>%
  summarise(median_MAE = median(MAE),
            ci_inf = quantile(MAE, 0.025),
            ci_sup = quantile(MAE, 0.975)) %>%
  ggplot(mapping = aes(x = nb_esn, y = median_MAE, ymin = ci_inf, ymax = ci_sup)) +
  geom_ribbon(fill = "grey") +
  geom_point() +
  geom_line() +
  theme_minimal() +
  labs(x = "Nb of Reservoir",
       y = "MAE (95% CI)")

ggsave(paste0(results_save, "plot_repeated_reservoir.pdf"),
       plot = plot_repeated_reservoir,
       width = 4,
       height = 4)

### 3) Graphical performance
df_individual_model <- df_all %>%
  filter(update == "No") %>%
  group_by(outcomeDate, model) %>%
  summarise(outcome = unique(outcome),
            hosp = unique(hosp),
            pred = median(pred),
            .groups = "drop") %>%
  mutate(outcomeDate = as.Date(outcomeDate)) %>%
  tidyr::pivot_wider(names_from = model, values_from = pred) %>%
  tidyr::pivot_longer(cols = -c(outcomeDate)) %>%
  mutate(name = factor(name,
                       levels = c("outcome",
                                  "hosp",
                                  "Elastic-net (RS)",
                                  "XGB (RS)",
                                  "Reservoir FS (RS)",
                                  "Reservoir no FS (GA)",
                                  "Reservoir FS (GA)")),
         name = forcats::fct_recode(name,
                                    "Observed" = "outcome",
                                    "Hosp t+14" = "hosp"))

plot_figure_performance_no_update <- df_individual_model %>%
  filter(outcomeDate >= as.Date("2021-04-12")) %>%
  filter(name %in% c("Hosp t+14",
                     "Observed",
                     "Reservoir FS (GA)",
                     "Elastic-net (RS)",
                     "XGB (RS)")) %>%
  ggplot(mapping = aes(x = outcomeDate, y = value, color = name)) +
  geom_line() +
  scale_color_manual(values = c("black", "darkgrey", "#005F73", "#94D2BD", "#AE2012")) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  labs(x = "Date",
       y = "Hospitalizations",
       color = "") +
  guides(color = guide_legend(ncol = 2))

ggsave(paste0(results_save, "plot_figure_performance_no_update.pdf"),
       plot = plot_figure_performance_no_update,
       width = 4,
       height = 5)

### 4) Supplementary all graphical performance
plot_figure_performance_all <- df_all %>%
  group_by(outcomeDate, model, update) %>%
  summarise(hosp = unique(hosp),
            outcome = unique(outcome),
            pred = median(pred),
            .groups = "drop") %>%
  mutate(outcomeDate = as.Date(outcomeDate),
         update = if_else(update == "Yes", "Hp updated each month", "Hp set at 2021-03-29")) %>%
  ggplot(mapping = aes(x = outcomeDate, y = pred, color = update)) +
  geom_line() +
  geom_line(mapping = aes(y = outcome, color = " Observed")) +
  geom_line(mapping = aes(y = hosp, color = "Hosp t+14")) +
  scale_color_manual(values = c("black", "darkgrey", "#AE2012", "#EE9B00")) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  labs(x = "Date",
       y = "Hospitalizations",
       color = "") +
  guides(color = guide_legend(ncol = 3)) +
  facet_wrap(model ~ ., ncol = 2)

ggsave(paste0(results_save, "plot_figure_performance_all.pdf"),
       plot = plot_figure_performance_all,
       width = 8)

########## HYPER-PARAMETERS #####
path_hp <- paste0(path_results, "hyperparameter")
ls_files_full <- list.files(path_hp, full.names = TRUE, recursive = TRUE)
names(ls_files_full) <- gsub(ls_files_full, pattern = paste0(path_hp, "/"), replacement = "")

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
                      mutate_all(as.character) %>%
                      tibble::rowid_to_column(var = "genetic_id")) %>%
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
                                   "Reservoir no FS (GA)",
                                   "Elastic-net (RS)",
                                   "XGB (RS)")),
         date = gsub(pattern = ".csv", x = date, replacement = ""),
         date = as.Date(date),
         date = if_else(is.na(date), as.Date("2021-03-01"), date),
         # last_used_observation = date + 2*14) %>%
         last_used_observation = date) %>%
  filter(last_used_observation < as.Date("2022-01-17"),
         value != 1000)

## save best model most important features
df_all_hp %>%
  group_by(last_used_observation) %>%
  filter(model == "Reservoir FS (GA)") %>%
  slice_min(value, n = 1) %>%
  select(c(last_used_observation, ends_with("_bin"))) %>%
  tidyr::pivot_longer(cols = -last_used_observation) %>%
  filter(value == "y") %>%
  select(last_used_observation, name) %>%
  mutate(name = gsub(name, replacement = "", pattern = "_bin")) %>%
  write.csv("results/best_features.csv", row.names = FALSE)

## get the best 40 by date
df_all_hp_best40 <- df_all_hp %>%
  group_by(model, last_used_observation) %>%
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
      group_by(last_used_observation) %>%
      mutate(rank = dense_rank(value),
             rank_factor = if_else(rank == 1, "Best", "Other")) %>%
      ungroup() %>%
      mutate(last_used_observation = as.factor(last_used_observation)) %>%
      arrange(desc(rank_factor)) %>%
      ggplot(mapping = aes(y = value, x = HP_value, fill = last_used_observation, size = rank_factor, color = rank_factor)) +
      geom_point(shape = 21) +
      facet_wrap(HP_name ~ ., scales = "free_x") +
      scale_fill_viridis_d(direction = -1) +
      scale_size_manual(values = c(4,2)) +
      scale_color_manual(values = c("red", "#AAAAAA00")) +
      scale_x_log10() +
      theme_minimal() +
      labs(x = "Hyperparameter value",
           y = "MAE",
           fill = "Update date",
           color = "",
           size = "",
           title = model_title) %>%
      return()
  })

plot_all_hp <- ggpubr::ggarrange(plotlist = ls_hpnum_plots, common.legend = TRUE, ncol = 2, nrow = 3, legend = "bottom")

ggsave(paste0(results_save, "plot_all_hp.pdf"),
       plot = plot_all_hp,
       width = 10, height = 12)

dfGeneticSinglIS_GA_visu <- df_all_hp %>%
  filter(model == "Reservoir FS (GA)") %>%
  group_by(date) %>%
  mutate(rank = dense_rank(value),
         rank_factor = case_when(rank <= 40 ~ "Best 40",
                                 rank <= 200 ~ "Best 200",
                                 rank > 200 ~ "Other"),
         rank_factor = factor(rank_factor,
                              levels = c("Best 200",
                                         "Best 40",
                                         "Other"))) %>%
  ungroup() %>%
  select(all_of(c("rank_factor", "genetic_id", "last_used_observation", "value", "spectral_radius", "leaking_rate", "input_scaling", "ridge"))) %>%
  tidyr::pivot_longer(cols = c("spectral_radius", "leaking_rate", "input_scaling", "ridge"),
                      values_to = "HP_value", names_to = "HP_name") %>%
  mutate(value = if_else(value > 50, 50, value))

plot_genetic_algo <- ggplot(data = dfGeneticSinglIS_GA_visu,
       mapping = aes(x = HP_value, y = value, color = genetic_id)) +
  geom_point() +
  # scale_color_brewer() +
  scale_color_viridis_c(direction = -1, option = "C", begin = 0.3) +
  labs(color = "Genetic individual") +
  ggnewscale::new_scale_color() +
  geom_point(data = dfGeneticSinglIS_GA_visu %>%
               filter(rank_factor != "Other") %>%
               arrange(rank_factor), 
             aes(color = rank_factor)) +
  scale_color_manual(values = c("black", "green")) +
  facet_grid(last_used_observation ~ HP_name, scales = "free_x") +
  scale_y_log10() +
  scale_x_log10(breaks = c(1e-10, 1e-5, 1, 1e5),
                labels = c("-10", "-5", "1", "5")) +
  theme_minimal() +
  theme(strip.text.y = element_text(angle = 0),
        legend.position = "bottom") +
  labs(x = "Log(Hyperparameter value)",
       y = "MAE",
       color = "")

dfGeneticSinglIS_GA_visu %>%
  filter(last_used_observation == "2021-12-30", rank_factor == "Best 40") %>%
  group_by(HP_name) %>%
  summarise(min(HP_value), max(HP_value))

ggsave(paste0(results_save, "plot_genetic_algo.pdf"),
       plot = plot_genetic_algo,
       useDingbats = TRUE,
       width = 5)

##### show categorical hyperparameters
categorical_features <- grep(colnames(df_all_hp_best40), pattern = "_bin$", value = TRUE)
df_all_hp_best40_categorical <- df_all_hp_best40 %>%
  select(all_of(c("job_id", "model", "last_used_observation", "value", categorical_features))) %>%
  tidyr::pivot_longer(cols = categorical_features,
                      values_to = "HP_value",
                      names_to = "HP_name") %>%
  na.omit() %>%
  group_by(model, last_used_observation, HP_name) %>%
  summarise(freq_select = mean(HP_value == "y"),
            .groups = "drop")

ls_hpcat_plots <- list()
df_all_hp_best40_categorical %>%
  mutate(HP_name = FctCleanFeaturesName(HP_name)) %>%
  group_by(model) %>%
  group_split() %>%
  lapply(FUN = function(df_i){
    
    model_title <- df_i$model %>% unique() %>% as.character
    
    vec_features_i <- df_i %>%
      filter(last_used_observation == min(last_used_observation)) %>%
      slice_max(freq_select, n = 10) %>%
      pull(HP_name)
    
    plot_importance <- df_i %>%
      filter(HP_name %in% vec_features_i) %>%
      ggplot(mapping = aes(x = last_used_observation, color = freq_select, y = HP_name)) +
      geom_point() +
      scale_color_viridis_c(direction = -1) +
      theme_minimal() +
      labs(x = "Update date",
           y = "",
           color = "Freq. select",
           title = model_title) +
      theme(legend.position = "bottom")
    
    ls_hpcat_plots[[model_title]] <<- plot_importance
    return()
  })

##### elastic-net and xgb importance when updating hyperparameter monthly for comparison
list_importance_enet_xgb <- list("Elastic-net" = paste0(path_results, "importance/", "enet_pred_RS_importance_combined.csv"),
                                 "XGBoost" = paste0(path_results, "importance/", "xgb_pred_RS_importance_combined.csv"))

lapply(names(list_importance_enet_xgb),
       function(x){
         df_imp_enet <- data.table::fread(list_importance_enet_xgb[[x]]) %>%
           mutate(last_used_observation = hp_date + 14) %>%
           group_by(outcomeDate) %>%
           slice_max(last_used_observation) %>%
           group_by(last_used_observation, features) %>%
           summarise(importance = mean(importance),
                     .groups = "drop")
         
         best_enet_features_first_date <- df_imp_enet %>%
           slice_min(last_used_observation) %>%
           slice_max(abs(importance), n = 10) %>%
           pull(features)
         
         color_lab <- ifelse(x == "XGBoost", "Mean Gain", "Mean Beta")
         
         plot_importance <- df_imp_enet %>%
           filter(features %in% best_enet_features_first_date) %>%
           mutate(features = FctCleanFeaturesName(features),
                  features = factor(features),
                  features = forcats::fct_reorder(features, abs(importance))) %>%
           ggplot(mapping = aes(x = last_used_observation, color = importance, y = features)) +
           geom_point() +
           scale_color_viridis_c(direction = -1) +
           theme_minimal() +
           labs(x = "Update date",
                y = "",
                color = color_lab,
                title = x) +
           theme(legend.position = "bottom")
         
         ls_hpcat_plots[[x]] <<- plot_importance
         return()
       })

ggsave(paste0(results_save, "plot_feature_imp_RCGA.pdf"),
       plot = ls_hpcat_plots$`Reservoir FS (GA)`,
       width = 6, height = 5)


plot_feature_imp <- ggpubr::ggarrange(plotlist = ls_hpcat_plots, ncol = 2, nrow = 3, legend = "bottom")

ggsave(paste0(results_save, "plot_feature_imp.pdf"),
       plot = plot_feature_imp,
       width = 12, height = 12)

############# explore why hp update did not work for GA
df_leaking_rate_RCGA <- df_all_hp %>%
  filter(model == "Reservoir FS (GA)") %>%
  mutate(hp_date = date + 14) %>%
  select(job_id, leaking_rate, hp_date) %>% 
  distinct()

df_by_lr <- df_all %>%
  filter(model == "Reservoir FS (GA)", update == "No") %>%
  mutate(job_id = gsub(x = trial, pattern = "^trial_|_train365$", replacement = ""),
         hp_date = as.Date(hp_date)) %>%
  left_join(df_leaking_rate_RCGA, by = c("job_id", "hp_date")) %>%
  mutate(leaking_rate = factor(leaking_rate < 1e-2,
                               levels = c(T, F),
                               labels = c("< 1e-2", "> 1e-2")),
         outcomeDate = as.Date(outcomeDate))

df_perf_by_lr <- df_by_lr %>%
  mutate(outcome = if_else(outcome < 10, 10, outcome),
         pred = if_else(pred < 10, 10, pred),
         hosp = if_else(hosp < 10, 10, hosp)) %>%
  group_by(outcomeDate, model, leaking_rate, update) %>%
  summarise(outcome = unique(outcome),
            hosp = unique(hosp),
            pred = median(pred),
            n = n(),
            .groups = "drop") %>%
  mutate(AE = abs(pred - outcome),
         RE = AE/outcome,
         baseline_AE = abs(hosp - outcome),
         AE_baseline = AE - baseline_AE,
         RE_baseline = AE/baseline_AE) %>%
  group_by(model, update, leaking_rate, n) %>%
  summarise(AE = mean(AE, na.rm = T),
            AE_baseline = mean(AE_baseline, na.rm = T),
            RE = median(RE, na.rm = TRUE),
            RE_baseline = median(RE_baseline, na.rm = TRUE),
            .groups = "drop")

df_perf_by_lr %>% knitr::kable(format = "latex", booktabs = TRUE, digits = 2)

plot_RCGA_noupdate_by_lr <- df_by_lr %>%
  select(outcomeDate, outcome, hosp, pred, leaking_rate) %>%
  tidyr::pivot_wider(values_from = pred, names_from = leaking_rate, values_fn = median, names_prefix = "leaking rate ") %>%
  tidyr::pivot_longer(cols = -outcomeDate) %>%
  mutate(name = factor(name,
                       levels = c("outcome",
                                  "hosp",
                                  "leaking rate < 1e-2",
                                  "leaking rate > 1e-2"),
                       labels = c("Observed",
                                  "Hosp t+14",
                                  "leaking rate < 1e-2",
                                  "leaking rate > 1e-2"))) %>%
  ggplot(mapping = aes(x = outcomeDate, y = value, color = name)) +
  geom_line() +
  scale_color_manual(values = c("black", "darkgrey", "#FB8500", "#219EBC")) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  labs(x = "Date",
       y = "Hospitalizations",
       color = "") +
  guides(color = guide_legend(ncol = 2))

ggsave(paste0(results_save, "plot_RCGA_noupdate_by_lr.pdf"),
       plot = plot_RCGA_noupdate_by_lr,
       height = 4,
       width = 8)

############# computing time and number of hp
df_time_hp <- df_all_hp %>%
  slice_min(date) %>%
  filter(model %in% c("Reservoir FS (RS)",
                      "Elastic-net (RS)",
                      "XGB (RS)")) %>%
  mutate(nb_features = select(., ends_with("_bin")) %>% mutate_all(.funs = function(x) x == "y") %>% rowSums()) %>%
  select(model, time_seconds, all_of(numeric_hp), nb_features) %>%
  tidyr::pivot_longer(cols = -c("model", "time_seconds")) %>%
  na.omit()

read.csv("data_obfuscated/20210329.csv") %>%
  select(-outcome, -outcomeDate, -START_DATE) %>%
  dim()

plot_time_hp <- df_time_hp %>%
  ggplot(mapping = aes(x = value, y = time_seconds)) +
  geom_point(size = 0.01) +
  scale_x_log10() +
  facet_wrap(model ~ name, scales = "free", ncol = 3) +
  theme_minimal()

df_time_hp %>%
  filter(name == "nb_features") %>%
  reframe(quantile(value, probs = c(0.25, 0.5, 0.75)))

ggsave("results/figures/plot_time_hp.pdf",
       plot = plot_time_hp,
       useDingbats = TRUE,
       height = 7,
       width = 7)

df_all_hp %>%
  slice_min(date) %>%
  group_by(model) %>%
  summarise(q1 = quantile(time_seconds, 0.25),
            median = quantile(time_seconds, 0.5),
            q3 = quantile(time_seconds, 0.75))

############# reservoir vs raw features importance 
df_imp_reservoir <- data.table::fread("results/importance/GeneticSingleIs_GA_importance_combined.csv") %>%
  mutate(last_used_observation = hp_date + 14) %>%
  slice_min(last_used_observation) %>%
  group_by(outcomeDate, trial) %>%
  mutate(rank = dense_rank(desc(abs(importance)))) %>%
  filter(!grepl(features, pattern = "reservoir")) %>%
  group_by(last_used_observation, features, outcomeDate) %>%
  summarise(importance = mean(rank),
            .groups = "drop") %>%
  group_by(outcomeDate) %>%
  mutate(rank_among_raws = dense_rank(importance))

plot_reservoir_vs_input_importance <- df_imp_reservoir %>%
  filter(rank_among_raws %in% c(1, 10, 50, 100, 200)) %>%
  mutate(rank_among_raws = factor(rank_among_raws,
                                  levels = c(1, 10, 50, 100, 200),
                                  labels = c("1st", "10th", "50th", "100th", "200th"))) %>%
  ggplot(mapping = aes(x = outcomeDate, y = importance, group = rank_among_raws, color = rank_among_raws)) +
  geom_line() +
  scale_color_viridis_d(direction = -1) +
  scale_y_reverse(limits = c(700,1),
                  breaks = c(1, 50, seq(100, 700, by = 100)),
                  labels = c("1st", "50th", paste0(seq(100, 700, by = 100), "th"))) +
  theme_minimal() +
  labs(y = "Importance rank according to output layer",
       x = "Date",
       color = "Rank among the input layer") +
  theme(legend.position = "bottom") +
  guides(color=guide_legend(nrow=2,byrow=TRUE))

ggsave(paste0(results_save, "plot_reservoir_vs_input_importance.pdf"),
       plot = plot_reservoir_vs_input_importance,
       width = 5, height = 5)

##### number of parameters
ls_importance <- list(RCGA = "results/importance/GeneticSingleIs_GA_importance_combined.csv",
                      Enet = "results/importance/enet_pred_RS_importance_combined.csv",
                      XGB = "results/importance/xgb_pred_RS_importance_combined.csv")

df_nb_hp <- lapply(ls_importance,
       FUN = function(path_i){
         path_i %>%
           data.table::fread() %>%
           slice_max(hp_date) %>%
           slice_max(outcomeDate) %>%
           select(nb_param) %>%
           reframe(min(nb_param), max(nb_param))
       }) %>%
  bind_rows(.id = "model")
