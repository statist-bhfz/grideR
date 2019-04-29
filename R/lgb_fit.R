#' Wrapper for lgb.train.
#'
#' Fit and evaluate lightgbm model with data.table as input data.
#' Model are trained (including all preprocessing steps) on train part and
#' evaluated on validation part according to \code{split} indicator variable.
#'
#' @param data data.table with all input data.
#' @param target Target variable name (character).
#' @param split Indicator variable with 1 corresponds to observations in validation dataset.
#' @param preproc_fun Preprocessing function which takes data.table \code{data}+\code{split}
#' as input and returns processed data.table with same \code{target} and \code{split} columns.
#' @param params 1-row data.table with all hyperparameters.
#' @param args List with parameters unchangeable during tuning.
#' @param metrics Vector of metric functions names.
#' @param return_val_preds If \code{TRUE}, predictions for validation data 
#' will be returned. 
#' @param return_model_obj If \code{TRUE}, model object will be returned.
#' @param train_on_all_data If \code{TRUE}, model will be fitted on all data
#' (without train/validation split) and model object will be returned.
#' @param ... Other parameters for \code{kgb.train()}.
#'
#' @return data.table with optimal number of iterations (if early stopping is used)
#' and all metrics calculated for validation part of the data. It also contains 
#' predictions for validation data if \code{return_val_preds = TRUE} and 
#' model object if \code{return_model_obj = TRUE}. 
#' If \code{train_on_all_data = TRUE}, only model object will be returned.
#'
#' @examples
#' # Input data
#' dt <- as.data.table(mtcars)
#' # data.table with resamples
#' splits <- resampleR::cv_base(dt, "hp")
#' # data.table with all hyperparameters
#'     lgb_grid <- CJ(
#'     learning_rate = 0.03, 
#'     metric = "rmse",
#'     num_leaves = 30,
#'     verbose = 1,
#'     subsample = 0.9,
#'     colsample_bytree = 0.8,
#'     random_state = 42,
#'     max_depth = c(3, 5, 7),
#'     lambda_l2 = 0.02,
#'     lambda_l1 = 0.004,
#'     bagging_fraction = 0.8,
#'     feature_fraction = 0.7,
#'     min_child_samples = 3,
#'     verbose = -1
#' )
#' # Non-tunable parameters for lightgbm
#' lgb_args <- list(
#'     nrounds = 1000,
#'     obj = "regression",
#'     early_stopping_rounds = 10,
#'     verbose = -1
#' )
#' # Dumb preprocessing function
#' # Real function will contain imputation, feature engineering etc.
#' # with all statistics computed on train folds and applied to validation fold
#' preproc_fun_example <- function(data) return(data[])
#' lgb_fit(data = dt,
#'         target = "hp",
#'         split = splits[, split_1],
#'         preproc_fun = preproc_fun_example,
#'         params = lgb_grid[1, ],
#'         args = lgb_args,
#'         metrics = c("rmse", "mae"),
#'         return_val_preds = TRUE)
#'
#' @details
#'
#'
#' @import data.table
#' @import checkmate
#' @import lightgbm
#' @export
lgb_fit <- function(data = data,
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
    
    data <- copy(data)[, split := split]
    data <- preproc_fun(data)
    
    if (train_on_all_data) {
        cols_to_drop <- c(target)
        dtrain <- lgb.Dataset(as.matrix(data[, .SD, .SDcols = -cols_to_drop]), 
                              label = data[, get(target)])
        rm(data)
        args <- c(lgb_args, 
                  list(params = as.list(params), 
                       data = dtrain))
        model <- do.call(lgb.train, args) 
        return(model)
    }
    
    val <- data[split == 1, ]
    
    cols_to_drop <- c(target, "split")
    
    dtrain <- lgb.Dataset(as.matrix(data[split == 0, .SD, .SDcols = -cols_to_drop]), 
                          label = data[split == 0, get(target)])
    
    dval <- lgb.Dataset(as.matrix(data[split == 1, .SD, .SDcols = -cols_to_drop]), 
                        label = data[split == 1, get(target)])
    
    rm(data)
    
    args <- c(lgb_args, 
              list(params = as.list(params), 
                   data = dtrain, 
                   valids = list(val = dval)))
    
    model <- do.call(lgb.train, args) 
    
    preds <- data.table(
        ground_truth = val[, get(target)],
        prediction = predict(model, as.matrix(val[, .SD, .SDcols = -cols_to_drop]))
    )
    
    res <- data.table(
        nrounds_best = model$best_iter
    )
    
    for (metric in metrics) {
        res[, (metric) := get(metric)(preds$ground_truth, preds$prediction)]
    }
    
    if (return_val_preds) res[, val_preds := .(list(preds[, prediction]))]
    if (return_model_obj) res[, model_obj := .(list(model))]
    
    return(res[])
}
