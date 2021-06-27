#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double sq_norm(double *x, int nu, double *y, double s_v, double hat_s_v, int j) {
  double sq_norm = 0;
  for (int i = 0; i < nu; ++i){
    sq_norm += pow((hat_s_v/s_v)*x[i+nu*j]-y[i],2);
  }
  return sq_norm;
}

double rho(double *y, double *eta, int nu, int N, double s_v, double hat_s_v){
  double rho_ = 0;
  for (int j = 0; j < N; ++j) {
    double exponential = exp(-0.5*sq_norm(eta, nu, y, s_v, hat_s_v, j)/(pow(hat_s_v,2)));
    rho_ += exponential/N;
  }
  return rho_;
}

double *gradient_rho(double * gradient, double *y, double *eta, int nu, int N, double s_v, double hat_s_v){
  for (int j = 0; j < N; ++j) {
    double exponential = exp(-0.5*sq_norm(eta, nu, y, s_v, hat_s_v, j)/(pow(hat_s_v,2)));
    for (int i = 0; i < nu; ++i) {
      gradient[i] += ((hat_s_v/s_v)*eta[j*nu+i]-y[i])*exponential/(N*(pow(hat_s_v,2)));
    }
  }
  return gradient;
}

//gcc -fPIC -O2 -c PLoM_C_library.c
//gcc -shared PLoM_C_library.o -o PLoM_C_library.so
