#' smoothDf
#'
#' @description Smooth double features of a dataframe
#'
#' @param df The dataframe
#' @param START_DATE The date column
#' @param skip_variables Variables to skip (default is NULL)
#' @param span_days The span (default is 14) it is then converted into span for loess by dividing by number of rows in df.
#' @param degree Degree of smoothing (default is 1)
#' @param DATE_colum The date column name
#'
#' @return A dataframe with the smoothed data
#' @export
smoothDf <- function(df,
                     DATE_colum = "START_DATE",
                     span_days = 14,
                     skip_variables = NULL,
                     degree = 1){

  ndays <- df %>% pull(all_of(DATE_colum)) %>% unique() %>% length()
  span <- span_days/ndays

  excludeTansf <- c(DATE_colum, skip_variables, "dblSTART_DATE")

  result <- df %>%
    mutate(across(where(is.double) & !any_of(excludeTansf),
                  function(x){
                    dblSTART_DATE <- as.numeric(get(DATE_colum))
                    loess(x ~ dblSTART_DATE,
                          data = .,
                          surface = "direct",
                          span = span,
                          degree = degree)$fitted
                  } ))

  return(result)
}
