#' Wrapper for catboost.train.
#'
#' Fit and evaluate catboost model with data.table as input data.
#' Model are trained (including all preprocessing steps) on train part and
#' evaluated on validation part according to \code{split} indicator variable.
#'
#' @param data data.table with all input data.
#' @param y Target variable name (character).
#' @param split Indicator variable with 1 corresponds to observations in validation dataset.
#' @param preproc_fun Preprocessing function which takes data.table \code{data}+\code{split}
#' as input and returns processed data.table with same \code{target} and \code{split} columns.
#' @param params 1-row data.table with all hyperparameters.
#' @param metrics Vector of metric functions names.
#' @param return_val_preds If \code{TRUE}, predictions for validation data 
#' will be returned. 
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
#' # data.table with all hyperparameters
#' catboost_grid <- CJ(
#'     iterations = 1000,
#'     learning_rate = 0.05,
#'     depth = c(8, 9),
#'     loss_function = "RMSE",
#'     eval_metric = "RMSE",
#'     random_seed = 42,
#'     od_type = 'Iter',
#'     # metric_period = 50,
#'     od_wait = 10,
#'     use_best_model = TRUE,
#'     logging_level = "Silent"
#' ) 
#' # Dumb preprocessing function
#' # Real function will contain imputation, feature engineering etc.
#' # with all statistics computed on train folds and applied to validation fold
#' preproc_fun_example <- function(data) return(data[])
#' catboost_fit(data = dt,
#'              target = "hp",
#'              split = splits[, split_1],
#'              preproc_fun = preproc_fun_example,
#'              params = catboost_grid[1, ],
#'              metrics = c("rmse", "mae"),
#'              return_val_preds = TRUE)
#'
#' @details
#'
#'
#' @import data.table
#' @import checkmate
#' @import catboost
#' @export
catboost_fit <- function(data = data,
                         target = target,
                         split = split,
                         preproc_fun = preproc_fun,
                         params = params,
                         metrics = metrics,
                         return_val_preds = FALSE,
                         ...) {
    
    assert_data_table(data)
    assert_integerish(split, len = data[, .N])
    assert_data_table(params)
    
    data <- copy(data)[, split := split]
    data <- preproc_fun(data)
    
    train <- data[split == 0, ]
    val <- data[split == 1, ]
    
    cols_to_drop <- c(target, "split")
    
    dtrain <- catboost.load_pool(train[, .SD, .SDcols = -cols_to_drop],
                                 label = train[, get(target)])
    
    dval <- catboost.load_pool(val[, .SD, .SDcols = -cols_to_drop],
                               label = val[, get(target)])
    
    model <- catboost.train(dtrain,
                            test_pool = dval,
                            params = as.list(params))
    
    preds <- data.table(
        ground_truth = val[, get(target)],
        prediction = catboost.predict(model, dval)
    )
    
    res <- data.table(
        nrounds_best = model$tree_count
    )
    for (metric in metrics) {
        res[, (metric) := get(metric)(preds$ground_truth, preds$prediction)]
    }
    if (return_val_preds) res[, val_preds := list(list(preds[, prediction]))]
    return(res[])
}