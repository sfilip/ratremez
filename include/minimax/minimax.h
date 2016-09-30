#ifndef MINIMAX_H
#define MINIMAX_H

#include "cheby.h"
#include "eigenvalue.h"
#include "ratbary.h"
#include "util.h"

void trialApproximant(mpfr::mpreal &h, std::vector<mpfr::mpreal> &fx,
                      std::vector<mpfr::mpreal> &x,
                      std::pair<std::size_t, std::size_t> &degree,
                      mp_prec_t prec = 165u);

void exchange(mpfr::mpreal& delta, std::vector<mpfr::mpreal> &x,
              std::function<mpfr::mpreal(mpfr::mpreal)> &f,
              std::pair<std::size_t, std::size_t> &degree, int Nmax,
              mp_prec_t prec = 165u);

#endif
