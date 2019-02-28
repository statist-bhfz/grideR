rmse <- function(ground_truth, prediction) {
    sqrt(mean((ground_truth - prediction)^2))
}

# 
mae <- function(ground_truth, prediction) {
    mean(abs(ground_truth - prediction))
}
