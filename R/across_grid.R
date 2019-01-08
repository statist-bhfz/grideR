#' Train and evaluate model across combinations of tunable hyperparameters.
#'
#' Fit and evaluate xgboost model with data.table as input data. 
#' Model are trained (including all preprocessing steps) on train part and
#' evaluated on validation part according to \code{split} indicator variable.
#'
#' @param data data.table with all input data.
#' @param y Target variable name (character).
#' @param split Indicator variable with 1 corresponds to observations in validation dataset.
#' @param preproc_fun Preprocessing function which takes data.table \code{data}+\code{split} 
#' as input and returns processed data.table with same \code{target} and \code{split} columns.
#' @param grid data.table with combinations of tunable hyperparameters in rows.
#' @param args List with parameters unchangeable during tuning.
#' @param metrics Vector of metric functions names.
#'
#' @return data.table with composed with \code{grid}, optimal numbers of iterions (implies  
#' that we use early stopping) and all metrics calculated for validation part of the data.
#'
#' @examples
#' # Input data
#' dt <- as.data.table(mtcars)
#' # data.table with resamples
#' splits <- resampleR::cv_base(data, target)
#' # data.table with tunable model hyperparameters
#' xgb_grid <- CJ(
#'     max_depth = c(6, 8),
#'     eta = 0.025,
#'     colsample_bytree = 0.9,
#'     subsample = 0.8,
#'     gamma = 0,
#'     min_child_weight = c(3, 5),
#'     alpha = 0,
#'     lambda = 1
#' )
#' # Non-tunable parameters for xgboost
#' xgb_args <- list(
#'     nrounds = 500,
#'     early_stopping_rounds = 10,
#'     booster = "gbtree",
#'     eval_metric = "rmse",
#'     objective = "reg:linear",
#'     verbose = 0
#' )
#' # Dumb preprocessing function
#' # Real function will contain imputation, feature engineering etc.
#' # with all statistics computed on train folds and applied to validation fold
#' preproc_fun_example <- function(data) return(data[])
#' across_grid(data = dt,
#'             target = "hp",
#'             split = splits[, split_1],
#'             fit_fun = xgb_fit,
#'             preproc_fun = preproc_fun_example,
#'             grid = xgb_grid,
#'             args = xgb_args,
#'             metrics = c("rmse", "mae"))
#'
#' @details
#' 
#'
#' @import data.table
#' @import checkmate
#' @import xgboost
#' @export
across_grid <- function(data,
                        target,
                        split,
                        fit_fun,
                        preproc_fun,
                        grid,
                        args,
                        metrics,
                        ...) {

    metrics <- lapply(seq_len(nrow(grid)),
                      function(i) fit_fun(data = data,
                                          target = target,
                                          split = split,
                                          preproc_fun = preproc_fun,
                                          params = grid[i, ],
                                          args = args,
                                          metrics = metrics))

    metrics <- rbindlist(metrics)

    data.table(grid, metrics)
}
