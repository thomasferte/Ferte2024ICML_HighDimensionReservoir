#' ComputeOutcome
#'
#' @description Compute the outcome on the given dataframe
#'
#' @param df Dataframe
#' @param OUTCOME The OUTCOME must be one of HOSP or IN_HOSP or HOSP_DERIV or IN_HOSP_in_COUNT_rolMean or RAW_HOSP
#' @param FORECAST The forecast time.
#' @param outcomeCol The outcome column. Default is 'CHU_HOSP'. Only works for OUTCOME in 'HOSP', 'HOSP_DERIV' or 'RAW_HOSP'.
#'
#' @return The dataframe with the outcome
#' @export
ComputeOutcome <- function(df,
                           OUTCOME,
                           FORECAST,
                           outcomeCol = "CHU_HOSP"){
  dfOutcomeRef <- ComputeOutcomeRef(df = df, OUTCOME = OUTCOME, FORECAST = FORECAST, outcomeCol = outcomeCol)
  dfOutcome <- ComputeOutcomeCustom(df = dfOutcomeRef, OUTCOME = OUTCOME, FORECAST = FORECAST)
  return(dfOutcome)
}
