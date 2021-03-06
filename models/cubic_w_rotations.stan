// These are some externally defined functions that do the RUS forward model.
//   That is, they take in elastic constants and produce resonance modes
functions {
  vector mech_init(int P, real X, real Y, real Z, real density);
  matrix mech_rotate(matrix C, vector q);
  vector mech_rus(int P, int N, vector lookup, matrix C);
}

// Input data
data {
  int<lower = 1> P; // Order of polynomials for Rayleigh-Ritz approx
  int<lower = 1> L; // This is a function of P :(
  int<lower = 1> N; // Number of resonance modes

  // Sample known quantities
  real<lower = 0.0> X;
  real<lower = 0.0> Y;
  real<lower = 0.0> Z;

  real<lower = 0.0> density;

  // Resonance modes
  vector[N] y;
}

transformed data {
  vector[L * L * 3 * 3 + L * L + L * L * 3 * 3 * 21 + L * L * 3 * 3] lookup;

  lookup = mech_init(P, X, Y, Z, density);
}

// Parameters to estimate
//
// Really I want to completely get rid of constraints on these
//   and move to weak priors. Constraints are bad business
parameters {
  real<lower = 0.0, upper = 4.0> c11;
  real<lower = 0.0, upper = 4.0> a;
  real<lower = 0.0, upper = 2.0> c44;
  real<lower = 0.0> invsigma; // we'll estimate measurement noise
  unit_vector[4] q; // rotation between sample & xtal axes
}

// Build a 6x6 stiffness matrix and rotate it
transformed parameters {
  real sigma;
  real c12;
  matrix[6, 6] C;

  sigma = 1 / invsigma;
  
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

  C = mech_rotate(C, q);
}

// This is the probabilistic model
model {
  vector[N] modes;
  // Specify a prior on noise level. Units are khz, we expect ~100-300hz in a good fit
  invsigma ~ gamma(3.0, 0.5);

  modes = mech_rus(P, N, lookup, C);

  // Resonance modes are normally distributed around what you'd expect from an RUS calculation
  y ~ normal(modes, sigma);

  print(y - modes);
}
