#' PrepareDataFromHyperparam
#'
#' @description Fonction to prepare data before training the model.
#'
#' @param span The span smoothing
#' @param model The statistical model ('enet' or 'poisson' or 'rf')
#' @param df The data
#' @param features The features to use
#' @param forecast The forecast time
#' @param date The date of prediction
#' @param outcomeCol The outcome column
#' @param rolderiv Should the ComputerolMeanMaxMinDeriv be computed
#' @param second_deriv Should the second derivative be computed (default is FALSE).
#'
#' @return The prepared dataframe
#' @export
PrepareDataFromHyperparam <- function(span = 7,
                                      model = "enet",
                                      df,
                                      features,
                                      forecast = 7,
                                      rolderiv = TRUE,
                                      date = as.Date("2020-10-01"),
                                      outcomeCol = "hosp",
                                      second_deriv = FALSE){
  outcome <- "RAW_HOSP"
  ### step1 : feature selection
  dfSelected <- df %>%
    select(any_of(features)) %>%
    ungroup()

  ## group by dep if dep exists
  if("dep" %in% colnames(df)){
    dfSelected <- dfSelected %>%
      group_by(dep)

  }

  ### step2: compute outcome and feature engineering (including smoothing)
  if(span == 0){
    dfOutcomeEngineeredTemp <- ComputeOutcome(df = dfSelected,
                                              OUTCOME = outcome,
                                              FORECAST = forecast,
                                              outcomeCol = outcomeCol)

    if(rolderiv){
      dfOutcomeEngineered <- dfOutcomeEngineeredTemp %>%
        ComputerolMeanMaxMinDeriv(df = .,
                                  second_deriv = second_deriv,
                                  excludeTansf = c("START_DATE",
                                                   "outcome",
                                                   "outcomeDate",
                                                   "outcomeRef",
                                                   "Vaccin_1dose",
                                                   "Population")) %>%
        ungroup()
    } else {
      dfOutcomeEngineered <- dfOutcomeEngineeredTemp %>%
        ungroup()
    }

  } else {
    dfOutcomeEngineered <- ComputeOutcomeRef(df = dfSelected,
                                             OUTCOME = outcome,
                                             FORECAST = forecast,
                                             outcomeCol = outcomeCol) %>%
      SmoothAndRolMinMaxMeanRespectDate(df = .,
                                        span_days = span,
                                        DATE = date,
                                        rolderiv = rolderiv,
                                        second_deriv = second_deriv,
                                        DATE_colum = "START_DATE",
                                        skip_variables = c("outcomeRef",
                                                           "outcomeDate",
                                                           "Vaccin_1dose",
                                                           "Population")) %>%
      ComputeOutcomeCustom(df = .,
                           OUTCOME = outcome,
                           FORECAST = forecast) %>%
      ungroup()
  }

  ## correct missing outcome Date
  dfOutcomeEngineered <- dfOutcomeEngineered %>%
    mutate(outcomeDate = if_else(is.na(outcomeDate), START_DATE + forecast, outcomeDate))

  ### step3 : transform character and factor features to dummies if model is using glmnet
  if(model %in% c("enet", "poisson", "esn")){
    vecFactoColumns <- dfSelected %>%
      select_if(.predicate = function(x) is.factor(x)|is.character(x)) %>%
      colnames()
    if(length(vecFactoColumns) >= 1){
      dfOutcomeEngineered <- fastDummies::dummy_columns(.data = dfOutcomeEngineered,
                                                        select_columns = vecFactoColumns,
                                                        remove_first_dummy = TRUE,
                                                        remove_selected_columns = FALSE)
    }
  }
  return(dfOutcomeEngineered)
}
