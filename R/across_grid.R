
across_grid <- function(data,
                        target,
                        split,
                        fit_fun,
                        preproc_fun,
                        grid,
                        args, ...) {

    metrics <- lapply(seq_len(nrow(grid)),
                      function(i) fit_fun(data = data,
                                          target = target,
                                          split = split,
                                          preproc_fun = preproc_fun,
                                          params = grid[i, ],
                                          args = args))

    metrics <- rbindlist(metrics)

    data.table(grid, metrics)
}


dt <- as.data.table(mtcars)
# data.table with resamples
splits <- cv_base(data, target)
# data.table with model hyperparameters
xgb_grid <- CJ(
    max_depth = c(6, 8),
    eta = 0.025,
    colsample_bytree = 0.9,
    subsample = 0.8,
    gamma = 0,
    min_child_weight = c(3, 5),
    alpha = 0,
    lambda = 1
)
# Non-tuned parameters for xgboost
xgb_args <- list(
    nrounds = 500,
    early_stopping_rounds = 10,
    booster = "gbtree",
    eval_metric = "rmse",
    objective = "reg:linear",
    verbose = 0
)
# Dumb preprocessing function
# Real function will contain imputation, feature engineering etc.
# with all statistics computed on train folds and applied to validation fold
preproc_fun_example <- function(data) {
    return(data[])
}
xgb_fit(data = dt,
        target = "hp",
        split = splits[, split_1],
        preproc_fun = preproc_fun_example,
        params = xgb_grid[1, ],
        args = xgb_args)

across_grid(data = dt,
            target = "hp",
            split = splits[, split_1],
            fit_fun = xgb_fit,
            preproc_fun = preproc_fun_example,
            grid = xgb_grid,
            args = xgb_args)
