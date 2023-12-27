#' ComputeOutcomeCustom
#'
#' @description Compute the custom outcome on the given dataframe
#'
#' @param df Dataframe
#' @param OUTCOME The OUTCOME must be one of HOSP or IN_HOSP
#' @param FORECAST The forecast time.
#'
#' @return The dataframe with the outcome
#' @export
ComputeOutcomeCustom <- function(df,
                                 OUTCOME,
                                 FORECAST){
  if(OUTCOME == "HOSP"){
    dfOutcome <- df %>%
      mutate(outcome = outcomeRef - CHU_HOSP)
  } else if(OUTCOME %in% c("IN_HOSP", "IN_HOSP_in_COUNT_rolMean", "RAW_HOSP")){
    dfOutcome <- df %>%
      mutate(outcome = outcomeRef)
  } else if(OUTCOME == "HOSP_DERIV"){
    dfOutcome <- df %>%
      mutate(outcome = outcomeRef - CHU_HOSP - FORECAST*CHU_HOSP_rolDeriv7)
  } else{
    stop("OUTCOME is unknown")
  }
  return(dfOutcome)
}
