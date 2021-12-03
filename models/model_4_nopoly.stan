data { 
  int <lower=0> num_continents;
  int <lower=0> n_train;
  matrix[n_train, num_continents] conts_train;
  vector[n_train] score_train;
  int <lower=0> n_test;
  matrix[n_test, num_continents] conts_test;
  vector[n_test] score_test;
  }
  
parameters {
  row_vector[num_continents] b_conts;
  real intercept;
  real<lower=0> sigma; // stdev
}

model {
  
  real mu;
  // priors
  b_conts ~ cauchy(0,1);
  // model
  
  for (i in 1:n_train) {
    
    mu = sum(conts_train[i] .* b_conts) +
    intercept;
    // model
    score_train[i] ~ normal(mu, sigma);
  }
}

generated quantities {
  vector[n_train] score_pred_train;
  real mse_train = 0;
  vector[n_test] score_pred_test;
  real mse_test = 0;
  vector [n_train] log_lik;
  
  // in sample mse
  for (i in 1:n_train) {
    // calculate terms
    real x;
    x = sum(conts_train[i] .* b_conts) +
    intercept;
    // sum polynomial terms together
    score_pred_train[i] = x;
    
    // mse calculation
    mse_train = mse_train + square(score_train[i] - score_pred_train[i]);
    log_lik[i] = normal_lpdf(score_train[i] | x, sigma);
  }
  // final mse division by n
  mse_train = mse_train / n_train;
  
  // out of sample mse
  for (i in 1:n_test) {
    real x;
    x = sum(conts_test[i] .* b_conts) +
    intercept;
    // sum polynomial terms together
    score_pred_test[i] = x;
    
    // mse calculation
    mse_test = mse_test + square(score_test[i] - score_pred_test[i]);
  }
  // final mse division by n
  mse_test = mse_test / n_test;
}

