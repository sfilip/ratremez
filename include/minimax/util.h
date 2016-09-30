/**
 * @file util.h
 * @author Silviu Filip
 * @date 12 March 2015
 * @brief Utility header file containing common includes used by the rest of the
 * library
 *
 */

//    firpm_ld
//    Copyright (C) 2015  S. Filip
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.

//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.

//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>

#ifndef UTIL_H_
#define UTIL_H_

#include "../mpreal.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <functional>
#include <gmp.h>
#include <iostream>
#include <mpfr.h>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

typedef std::pair<mpfr::mpreal, mpfr::mpreal> Interval;

struct RatFunc {
  std::vector<mpfr::mpreal> coeffs;
  std::size_t nn, dd;

  RatFunc(std::vector<mpfr::mpreal> &num, std::vector<mpfr::mpreal> &denom)
      : coeffs(num.size() + denom.size() - 1u), nn(num.size()),
        dd(denom.size()) {
    for (std::size_t j{0u}; j < nn; ++j)
      coeffs[j] = num[j] / denom[0u];
    for (std::size_t j{1u}; j < dd; ++j)
      coeffs[j + nn - 1u] = denom[j] / denom[0u];
  }

  RatFunc(std::vector<mpfr::mpreal> &mCoeffs, const std::size_t n,
          const std::size_t d)
      : coeffs(mCoeffs), nn(n), dd(d) {}

  mpfr::mpreal operator()(mpfr::mpreal x) const {
    mpfr::mpreal sumn = 0.0, sumd = 0.0;
    for (int j{(int)nn - 1}; j >= 0; --j)
      sumn = mpfr::fma(sumn, x, coeffs[j]);
    for (int j{(int)(nn + dd) - 2}; j >= (int)nn; --j)
      sumd = mpfr::fma(sumd, x, coeffs[j]);

    return sumn / (sumd * x + 1);
  }
};

#endif /* UTIL_H_ */
