#' rolDeriv
#'
#' @description Deriv over last lagDays days
#'
#' @param x feature
#' @param lagDays The deepness of the derivative
#'
#' @return The transformed feature
#' @export
rolDeriv = function(x, lagDays){
  (x-lag(x, n = (lagDays-1)))/lagDays
}

#' rolDeriv2
#'
#' @description Second deriv over last lagDays days
#'
#' @param x feature
#' @param lagDays The deepness of the derivative
#'
#' @return The transformed feature
#' @export
rolDeriv2 = function(x, lagDays){
  (x - 2*lag(x, n = (lagDays-1)) + lag(x, n = 2*(lagDays-1))) / lagDays
}
