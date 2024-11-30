
data {
  int<lower=1> n_games;
  int<lower=1> n_teams;
  

  array[n_games] int<lower=1, upper=n_teams> winner_id;
  array[n_games] int<lower=1, upper=n_teams> loser_id;
  
  int<lower=1> n_pred;
  array[n_pred] int<lower=1, upper=n_teams> pred_t1_id;
  array[n_pred] int<lower=1, upper=n_teams> pred_t2_id;
}

parameters {
  vector[n_teams] team_ratings;
  real<lower = 0> sigma;
}


model {
  sigma ~ normal(0, 1);
  team_ratings ~ normal(0, sigma);
  
  1 ~ bernoulli_logit(team_ratings[winner_id] - team_ratings[loser_id]);
}

generated quantities {
  vector[n_pred] t1_pred;
  vector[n_pred] t2_pred;
  for(i in 1:n_pred){
    t1_pred[i] = exp(team_ratings[pred_t1_id[i]]) / (exp(team_ratings[pred_t1_id[i]]) + exp(team_ratings[pred_t2_id[i]]));
    t2_pred[i] = exp(team_ratings[pred_t2_id[i]]) / (exp(team_ratings[pred_t1_id[i]]) + exp(team_ratings[pred_t2_id[i]]));
  }
}