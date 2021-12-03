data { 
  
  int <lower=0> n_train;
  vector[n_train] score_train;
  vector[n_train] economy_train;
  vector[n_train] corruption_train;
  int <lower=0> n_test;
  vector[n_test] score_test;
  vector[n_test] economy_test;
  vector[n_test] corruption_test;
  }
  
parameters {
  real b_economy;     // poly terms
  real b_corruption;     // poly terms
  real intercept;
  real<lower=0> sigma; // stdev
}

model {
  
  real mu;
  // priors
  b_economy ~ cauchy(0, 1);
  b_corruption ~ cauchy(0, 1);
  // model
  
  for (i in 1:n_train) {
    
    mu = economy_train[i]*b_economy + corruption_train[i]*b_corruption + intercept;
    // model
    score_train[i] ~ normal(mu, sigma);
  }
}

generated quantities {
  vector[n_train] score_pred_train;
  real mse_train = 0;
  vector[n_test] score_pred_test;
  real mse_test = 0;
  vector[n_train] log_lik;
  
  // in sample mse
  for (i in 1:n_train) {
    // calculate terms
    
    
    // sum polynomial terms together
    score_pred_train[i] = economy_train[i]*b_economy + corruption_train[i]*b_corruption + intercept;;
    
    // mse calculation
    mse_train = mse_train + square(score_train[i] - score_pred_train[i]);
  }
  // final mse division by n
  mse_train = mse_train / n_train;
  
  // out of sample mse
  for (i in 1:n_test) {

    
    // sum polynomial terms together
    real mu = economy_test[i]*b_economy + corruption_test[i]*b_corruption + intercept;
    score_pred_test[i] = mu;
    
    // mse calculation
    mse_test = mse_test + square(score_test[i] - score_pred_test[i]);
    log_lik[i] = normal_lpdf(score_train[i] | mu, sigma);
  }
  // final mse division by n
  mse_test = mse_test / n_test;
}
