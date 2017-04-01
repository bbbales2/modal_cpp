functions {
  vector mech(int N, vector dp, matrix pv, real c11, real a, real c44);
}

data {
  int<lower = 0> N;
  //int<lower = 0> M;
  int<lower = 0> L;

  vector[L * L * 3 * 3] dp;
  matrix[L, L] pv;

  vector[N] y;
}

parameters {
  real<lower = 0.0> c11;
  real a;
  real c44;
  real<lower = 0.0> sigma;
}

model {
  vector[N] res;

  //c11 ~ normal(2.0, 0.5);
  //a ~ normal(1.0, 0.25);
  //c44 ~ normal(0.75, 0.5);

  // c11, a, c44 -- params
  // N -- number of resonance modes
  // lookup, S -- lookup tables
  res = mech(N, dp, pv, c11, a, c44);

  //for (i in 1:M)
  for (j in 1:N)
    y[j] ~ normal(res[j], sigma);
}
