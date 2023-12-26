#' ComputerolMeanMaxMinDeriv
#'
#' @description Function to compute rolMean, rolMin, rolMax and rolDeriv on double columns of a dataset
#'
#' @param df The dataset
#' @param excludeTansf The double column not to take into account.
#' @param date_column The date column. Default is 'START_DATE'.
#' @param second_deriv Should the second derivative be computed (default is FALSE).
#'
#' @return The dataframe with the transformed features
#' @export
ComputerolMeanMaxMinDeriv <- function(df,
                                      date_column = "START_DATE",
                                      excludeTansf = "START_DATE",
                                      second_deriv = FALSE){
  if(second_deriv){
    ls_fct_rolDeriv = list(rolDeriv3 = function(x) rolDeriv(x, 3),
                           rolDeriv7 = function(x) rolDeriv(x, 7),
                           rolDeriv10 = function(x) rolDeriv(x, 10),
                           rolDeriv14 = function(x) rolDeriv(x, 14),
                           rol2Deriv3 = function(x) rolDeriv2(x, 3),
                           rol2Deriv7 = function(x) rolDeriv2(x, 7),
                           rol2Deriv10 = function(x) rolDeriv2(x, 10),
                           rol2Deriv14 = function(x) rolDeriv2(x, 14))
  } else {
    ls_fct_rolDeriv = list(rolDeriv3 = function(x) rolDeriv(x, 3),
                           rolDeriv7 = function(x) rolDeriv(x, 7),
                           rolDeriv10 = function(x) rolDeriv(x, 10),
                           rolDeriv14 = function(x) rolDeriv(x, 14))
  }

  result <- df %>%
    arrange(date_column) %>%
    mutate(across(where(is.double) & !any_of(excludeTansf),
                  list(rolMean = rolMean,
                       rolMax = rolMax,
                       rolMin = rolMin))) %>%
    mutate(across(where(is.double) & !any_of(excludeTansf),
                  ls_fct_rolDeriv)) %>%
    dplyr::filter(across(.cols = -any_of(c("outcome", "outcomeDate", "outcomeRef")),
                         .fns = ~ !is.na(.x)))
  return(result)
}
