#' SmoothAndRolMinMaxMeanRespectDate
#'
#' @description Smooth dataframe and add rolDeriv features considering only data available at given DATE
#'
#' @param df Dataframe
#' @param DATE Date of availability
#' @param DATE_colum Date colum
#' @param skip_variables Variables to skip
#' @param span_days The span (default is 14) it is then converted into span for loess by dividing by number of rows in df.
#' @param rolderiv Should the ComputerolMeanMaxMinDeriv be computed
#' @param second_deriv Should the second derivative be computed (default is FALSE).
#'
#' @return A smoothed dataframe with the additional features
#' @export
SmoothAndRolMinMaxMeanRespectDate <- function(df,
                                              span_days = 14,
                                              DATE,
                                              rolderiv = TRUE,
                                              second_deriv = FALSE,
                                              DATE_colum = "START_DATE",
                                              skip_variables = c("outcome", "outcomeDate")){
  dftemp <- df %>%
    filter(START_DATE <= DATE) %>%
    smoothDf(DATE_colum = DATE_colum,
             span_days = span_days,
             df = .,
             skip_variables = skip_variables)

  if(rolderiv){
    dfSmooth <- dftemp %>%
      ComputerolMeanMaxMinDeriv(.,
                                excludeTansf = c(DATE_colum, skip_variables),
                                second_deriv = second_deriv)
  } else {
    dfSmooth <- dftemp
  }

  return(dfSmooth)
}
