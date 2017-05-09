functions {
  vector mech_init(int P, real X, real Y, real Z, real density);
  vector mech_rus(int P, int N, vector lookup, matrix C);
}

data {
  int<lower = 1> P; // Maximum order polynomials to use in RR solution
  int<lower = 1> L; // This is a function of P :(
  int<lower = 1> N; // Number of resonance modes

  real<lower = 0.0> X;
  real<lower = 0.0> Y;
  real<lower = 0.0> Z;

  real<lower = 0.0> density;
  
  vector[N] y;
}

transformed data {
  vector[L * L * 3 * 3 + L * L + L * L * 3 * 3 * 21 + L * L * 3 * 3] lookup;

  lookup = mech_init(P, X, Y, Z, density);
}

parameters {
  real<lower = 0.0, upper = 4.0> c11;
  real<lower = 0.5, upper = 4.0> a;
  real<lower = 0.25, upper = 2.0> c44;
  real<lower = 0.0> sigma;
}

transformed parameters {
  real c12;
  matrix[6, 6] C;
  
  c12 = -(c44 * 2.0 / a - c11);

  for (i in 1:6)
    for (j in 1:6)
      C[i, j] = 0.0;
        
  C[1, 1] = c11;
  C[2, 2] = c11;
  C[3, 3] = c11;
  C[4, 4] = c44;
  C[5, 5] = c44;
  C[6, 6] = c44;
  C[1, 2] = c12;
  C[1, 3] = c12;
  C[2, 3] = c12;
  C[3, 2] = c12;
  C[2, 1] = c12;
  C[3, 1] = c12;
}

model {
  sigma ~ normal(0, 2.0);

  y ~ normal(mech_rus(P, N, lookup, C), sigma);
}
