#ifndef RATBARY_H
#define RATBARY_H

#include "util.h"


void barycentricWeights(std::vector<mpfr::mpreal>& w,
        std::vector<mpfr::mpreal>& x, std::vector<mpfr::mpreal>& fx,
        std::pair<std::size_t, std::size_t>& deg, mp_prec_t prec = 165u);

void evaluateBary(mpfr::mpreal &result, const mpfr::mpreal& xVal,
        std::vector<mpfr::mpreal>& x, std::vector<mpfr::mpreal> &fx,
        std::vector<mpfr::mpreal>& w, mp_prec_t prec = 165ul);

void barycentricWeights(std::vector<double>& w,
        std::vector<double>& x, std::vector<double>& fx,
        std::pair<std::size_t, std::size_t>& deg);

void evaluateBary(double &result, double& xVal,
        std::vector<double>& x, std::vector<double> &fx,
        std::vector<double>& w);

#endif // RATBARY
