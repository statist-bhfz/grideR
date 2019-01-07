

xgb_fit <- function(data = data,
                    target = target,
                    split = split,
                    preproc_fun = preproc_fun,
                    params = params,
                    args = args,
                    metrics = metrics,
                    ...) {

    assert_data_table(data)
    assert_integerish(split, len = data[, .N])
    assert_data_table(params)
    assert_list(args)

    data <- copy(data)[, split := split]
    data <- preproc_fun(data)

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
                   watchlist = list(vas = dval),
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
    return(res[])
}
