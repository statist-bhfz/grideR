#' Wrapper for xgb.train.
#'
#' Fit and evaluate xgboost model with data.table as input data.
#' Model are trained (including all preprocessing steps) on train part and
#' evaluated on validation part according to \code{split} indicator variable.
#'
#' @param data data.table with all input data.
#' @param target Target variable name (character).
#' @param split Indicator variable with 1 corresponds to observations in validation dataset.
#' @param preproc_fun Preprocessing function which takes data.table \code{data}+\code{split}
#' as input and returns processed data.table with same \code{target} and \code{split} columns.
#' @param params 1-row data.table with tunable hyperparameters.
#' @param args List with parameters unchangeable during tuning.
#' @param metrics Vector of metric functions names.
#' @param return_val_preds If \code{TRUE}, predictions for validation data 
#' will be returned.
#' @param return_model_obj If \code{TRUE}, model object will be returned.
#' @param train_on_all_data If \code{TRUE}, model will be fitted on all data
#' (without train/validation split) and model object will be returned.
#' @param ... Other parameters for \code{xgb.train()}.
#'
#' @return data.table with optimal number of iterations (implies that we use early stopping)
#' and all metrics calculated for validation part of the data. It also contains 
#' predictions for validation data if \code{return_val_preds = TRUE}.
#'
#' @examples
#' # Input data
#' dt <- as.data.table(mtcars)
#' # data.table with resamples
#' splits <- resampleR::cv_base(dt, "hp")
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
#' xgb_fit(data = dt,
#'         target = "hp",
#'         split = splits[, split_1],
#'         preproc_fun = preproc_fun_example,
#'         params = xgb_grid[1, ],
#'         args = xgb_args,
#'         metrics = c("rmse", "mae"))
#'
#' @details
#'
#'
#' @import data.table
#' @import checkmate
#' @import xgboost
#' @export
xgb_fit <- function(data = data,
                    target = target,
                    split = split,
                    preproc_fun = preproc_fun,
                    params = params,
                    args = args,
                    metrics = metrics,
                    return_val_preds = FALSE,
                    return_model_obj = FALSE,
                    train_on_all_data = FALSE,
                    ...) {

    assert_data_table(data)
    if (!train_on_all_data) assert_integerish(split, len = data[, .N])
    assert_data_table(params)
    assert_list(args)

    data <- copy(data)[, split := split]
    data <- preproc_fun(data)
    
    if(train_on_all_data) {
        cols_to_drop <- c(target)
        dtrain <- xgb.DMatrix(
            data = as.matrix(data[, .SD, .SDcols = -cols_to_drop]),
            label = as.matrix(data[, get(target)])
        )
        args <- c(args,
                  list(params = as.list(params),
                       data = dtrain))
        model <- do.call(xgb.train, args)
        return(model)
    }
    
    train <- data[split == 0, ]
    val <- data[split == 1, ]
    
    cols_to_drop <- c(target, "split")
    
    dtrain <- xgb.DMatrix(
        data = as.matrix(train[, .SD, .SDcols = -cols_to_drop]),
        label = as.matrix(train[, get(target)])
    )

    dval <- xgb.DMatrix(
        data = as.matrix(val[, .SD, .SDcols = -cols_to_drop]),
        label = as.matrix(val[, get(target)])
    )

    args <- c(args,
              list(params = as.list(params),
                   watchlist = list(val = dval),
                   data = dtrain))

    model <- do.call(xgb.train, args)

    preds <- data.table(
        ground_truth = val[, get(target)],
        prediction = predict(model, dval)
    )

    res <- data.table(
        nrounds_best = model$best_iteration
    )
    
    for (metric in metrics) {
        res[, (metric) := get(metric)(preds$ground_truth, preds$prediction)]
    }
    
    if (return_val_preds) res[, val_preds := .(list(preds[, prediction]))]
    if (return_model_obj) res[, model_obj := .(list(model))]
    
    return(res[])
}
