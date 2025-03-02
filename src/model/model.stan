
data {
  int<lower=0> N;
  int<lower=0> n_teams;
  
  array[N] int<lower=1, upper=n_teams> home_teams;
  array[N] int<lower=1, upper=n_teams> away_teams;
  array[N] int<lower=0> home_goals;
  array[N] int<lower=0> away_goals;
  
  
  int<lower=0> N_new;
  array[N_new] int<lower=1, upper=n_teams> home_new;
  array[N_new] int<lower=1, upper=n_teams> away_new;
}


parameters {
  real mu;
  real is_home;
  
  vector[n_teams] att;
  vector[n_teams] def;
  
  real<lower=0> att_sigma;
  real<lower=0> def_sigma;
}

model {
  mu ~ normal(0, 1);
  is_home ~ normal(0, 1);
  
  att ~ normal(0, att_sigma);
  att_sigma ~ normal(0, 1);
  
  def ~ normal(0, def_sigma);
  def_sigma ~ normal(0, 1);
  
  
  home_goals ~ poisson_log(mu + is_home + att[home_teams] + def[away_teams]);
  away_goals ~ poisson_log(mu + att[away_teams] + def[home_teams]);
  
}

generated quantities{

  array[N_new] real home_rate;
  for (i in 1:N_new) {
    home_rate[i] = mu + is_home + att[home_new[i]] + def[away_new[i]];
  }

  array[N_new] real away_rate;  
  for (i in 1:N_new) {
    away_rate[i] = mu + att[away_new[i]] + def[home_new[i]];
  }

  array[N_new] real pred_home_goals = poisson_log_rng(home_rate);
  array[N_new] real pred_away_goals = poisson_log_rng(away_rate);
  
  //Probability the home team scores first in OT 
  array[N_new] real home_ot_win_prob;
  for(i in 1:N_new){
    home_ot_win_prob[i] = exp(home_rate[i]) / (exp(home_rate[i]) + exp(away_rate[i]));
  }
  
}

