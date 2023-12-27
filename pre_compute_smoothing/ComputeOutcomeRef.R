#' ComputeOutcomeRef
#'
#' @description Compute the outcome of reference on the given dataframe
#'
#' @param df Dataframe
#' @param OUTCOME The OUTCOME must be one of HOSP or IN_HOSP
#' @param FORECAST The forecast time.
#' @param outcomeCol The outcome column. Default is 'CHU_HOSP'. Only works for OUTCOME in 'HOSP', 'HOSP_DERIV' or 'RAW_HOSP'.
#'
#' @return The dataframe with the outcome
#' @export
ComputeOutcomeRef <- function(df,
                              OUTCOME,
                              FORECAST,
                              outcomeCol = "CHU_HOSP"){
  dfOutcomeDate <- df %>%
    arrange(START_DATE) %>%
    mutate(outcomeDate = lead(START_DATE, max(FORECAST)))

  if(OUTCOME %in% c("HOSP", "HOSP_DERIV", "RAW_HOSP")){
    if(length(FORECAST) == 1){
      dfOutcome <- dfOutcomeDate %>%
        mutate(outcomeRef = lead(get(outcomeCol), FORECAST))
    } else {

      ls_fct <- lapply(FORECAST,
                       FUN = function(forecast_i){
                         function(outcomeCol) lead(outcomeCol, forecast_i)
                       })
      names(ls_fct) <- paste0("outcomeRef_", FORECAST, "d")

      dfOutcome <- dfOutcomeDate %>%
        mutate(across(all_of(outcomeCol),
                      ls_fct,
                      .names = "{fn}")) %>%
        mutate(outcomeRef = purrr::pmap(select(., starts_with("outcomeRef")),
                                        c)) %>%
        select(-starts_with("outcomeRef_"))

    }

  } else if(OUTCOME == "IN_HOSP"){
    dfOutcome <- dfOutcomeDate %>%
      mutate(outcomeRef = lead(IN_HOSP_in_COUNT, FORECAST))
  }  else if(OUTCOME == "IN_HOSP_in_COUNT_rolMean"){
    dfOutcome <- dfOutcomeDate %>%
      mutate(temp_IN_HOSP_in_COUNT_rolMean = rolMean(IN_HOSP_in_COUNT),
             outcomeRef = lead(temp_IN_HOSP_in_COUNT_rolMean, FORECAST)) %>%
      select(-temp_IN_HOSP_in_COUNT_rolMean)
  } else{
    stop("OUTCOME is unknown")
  }
  return(dfOutcome)
}
