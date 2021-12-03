data { 
  int <lower=0> num_continents;
  int <lower=0> n_train;
  vector[n_train] score_train;
  vector[n_train] economy_train;
  vector[n_train] corruption_train;
  matrix[n_train, num_continents] conts_train;
  int <lower=0> n_test;
  vector[n_test] score_test;
  vector[n_test] economy_test;
  vector[n_test] corruption_test;
  matrix[n_test, num_continents] conts_test;
  }
  
parameters {
  real b_economy;
  real b_economy_2;
  real b_economy_3;// poly terms
  real b_corruption; 
  real b_corruption_2;
  real b_corruption_3;
  real b_inter_11;
  real b_inter_12;
  real b_inter_21;
  real b_inter_22;// poly terms
  real intercept;
  row_vector[num_continents] b_conts;
  real<lower=0> phi; // stdev
}
transformed parameters {
  real x_train;
  real x_test;
  vector [n_train] mu_train;
  vector[n_train] ni_train;
  vector[n_train] lambda_train;
  vector[n_test] mu_test;
  vector[n_test] ni_test;
  vector[n_test] lambda_test;
  for (i in 1:n_train) {
    x_train = pow(economy_train[i], 3)*b_economy_3+
    pow(economy_train[i], 2) * b_economy_2 + economy_train[i]*b_economy + 
    pow(corruption_train[i], 3)*b_corruption_3 +
    pow(corruption_train[i], 2)*b_corruption_2 +
    corruption_train[i]*b_corruption + 
    pow(economy_train[i], 2)*pow(corruption_train[i], 2) * b_inter_22 +
    pow(economy_train[i], 2)*pow(corruption_train[i], 1) * b_inter_21 +
    pow(economy_train[i], 1)*pow(corruption_train[i], 2) * b_inter_12 +
    pow(economy_train[i], 1)*pow(corruption_train[i], 1) * b_inter_11 +
    sum(conts_train[i] .* b_conts) +
    intercept;
    
    mu_train[i] = x_train;
  }
  for (i in 1:n_test) {
    x_test = pow(economy_test[i], 3)*b_economy_3+
    pow(economy_test[i], 2) * b_economy_2 + economy_train[i]*b_economy + 
    pow(corruption_test[i], 3)*b_corruption_3 +
    pow(corruption_train[i], 2)*b_corruption_2 +
    corruption_test[i]*b_corruption + 
    pow(economy_test[i], 2)*pow(corruption_test[i], 2) * b_inter_22 +
    pow(economy_test[i], 2)*pow(corruption_test[i], 1) * b_inter_21 +
    pow(economy_test[i], 1)*pow(corruption_test[i], 2) * b_inter_12 +
    pow(economy_test[i], 1)*pow(corruption_test[i], 1) * b_inter_11 +
    sum(conts_test[i] .* b_conts) +
    intercept;
    
    mu_test[i] = x_test;
  }
  
  
  ni_train = mu_train .* mu_train /phi;
  ni_test = mu_test .* mu_test /phi;
  lambda_train = mu_train / phi;
  lambda_test = mu_test /phi;
}


model {
  
  real mu;
  // priors
  b_economy ~ cauchy(0, 1);
  b_economy_2 ~ cauchy(0,1);
  b_economy_3 ~ cauchy(0,1);
  b_corruption ~ cauchy(0, 1);
  b_corruption_2 ~ cauchy(0,1);
  b_corruption_3 ~ cauchy(0,1);
  b_inter_12 ~ cauchy(0,1);
  b_inter_21 ~ cauchy(0,1);
  b_inter_11 ~ cauchy(0,1);
  b_inter_22 ~ cauchy(0,1);
  b_conts ~ cauchy(0,1);
  // model
  
  for (i in 1:n_train) {
    // model
    score_train[i] ~ gamma(ni_train, lambda_train);
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
    // sum polynomial terms together
    score_pred_train[i] = ni_train[i];
    
    // mse calculation
    mse_train = mse_train + square(score_train[i] - score_pred_train[i]);
    log_lik[i] = normal_lpdf(score_train[i] | ni_train[i], lambda_train);
  }
  // final mse division by n
  mse_train = mse_train / n_train;
  
  // out of sample mse
  for (i in 1:n_test) {
    
    // sum polynomial terms together
    score_pred_test[i] = ni_test[i];
    
    // mse calculation
    mse_test = mse_test + square(score_test[i] - score_pred_test[i]);
  }
  // final mse division by n
  mse_test = mse_test / n_test;
}

