########################### load packages #########################
library(dplyr)

nb_cores = parallel::detectCores()-2

file_path <- list.files(path = "pre_compute_smoothing",
                        full.names = TRUE,
                        recursive = TRUE)

invisible(lapply(file_path, source))

########################### load data #########################
dfEDS <- readRDS(file = "data/df_obfuscated.rds")

########################### time windows #########################
forecast <- 14
vecDates <- dfEDS %>%
  # only even dates
  mutate(bool = START_DATE %>% as.numeric() %% 2 == 1) %>%
  filter(bool) %>%
  #
  select(START_DATE) %>%
  arrange(START_DATE) %>%
  slice(-c(1:28)) %>%
  pull(START_DATE)

########################### precompute data #####################
print(nb_cores)
cl <- parallel::makeCluster(nb_cores)

parallel::clusterExport(cl,
                        ls(),
                        envir = environment())
print("data")
res <- parallel::parLapply(cl = cl,
                           X = vecDates,
                           fun = function(date_i){
                             library(dplyr)
                             dfOutcomeEngineeredTrain <- PrepareDataFromHyperparam(span = 21,
                                                                                   model = "esn",
                                                                                   df = dfEDS,
                                                                                   features = colnames(dfEDS),
                                                                                   forecast = forecast,
                                                                                   rolderiv = TRUE,
                                                                                   second_deriv = TRUE,
                                                                                   date = date_i,
                                                                                   outcomeCol = "hosp") %>%
                               # remove redundant features
                               select(!matches("_rolMax|_rolMin|_rolMean|_rol2Deriv[3,10,14]|_rolDeriv[3,10,14]")) %>%
                               select(-c("WEEKDAY", "outcomeRef", "Majority_variant"))
                             
                             if(nrow(dfOutcomeEngineeredTrain) > (30+14)){
                               write.csv(x = dfOutcomeEngineeredTrain,
                                         row.names = FALSE,
                                         file = paste0("data_obfuscated/",
                                                       gsub(date_i, pattern = "-", replacement = ""),
                                                       ".csv"))
                             }
                             
                             if(date_i == max(vecDates)){
                               vec_features <- dfOutcomeEngineeredTrain %>%
                                 select(-c(START_DATE, outcome, outcomeDate)) %>%
                                 colnames() %>%
                                 paste0(collapse = ', ', '"', ., '"') %>%
                                 paste0("[", ., "]") %>%
                                 writeLines("data/allfeatures")
                             }
                             
                             return()
                           })
parallel::stopCluster(cl)

