#include "minimax/ratbary.h"

/** Eigen matrix container for mpfr::mpreal values */
typedef Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, Eigen::Dynamic> MatrixXm;
/** Eigen vector container for mpfr::mpreal values */
typedef Eigen::Matrix<std::complex<mpfr::mpreal>, Eigen::Dynamic, 1> VectorXcm;

/** Eigen matrix container for mpfr::mpreal values */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
/** Eigen vector container for mpfr::mpreal values */
typedef Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> VectorXcd;

void barycentricWeights(std::vector<mpfr::mpreal> &w,
                        std::vector<mpfr::mpreal> &x, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);
  w.resize(x.size());

  if (x.size() > 500u) {
    for (std::size_t i = 0u; i < x.size(); ++i) {
      mpfr::mpreal one = 1;
      mpfr::mpreal denom = 0.0;
      mpfr::mpreal xi = x[i];
      for (std::size_t j = 0u; j < x.size(); ++j) {
        if (j != i) {
          denom += mpfr::log(((xi - x[j] > 0) ? (xi - x[j]) : (x[j] - xi)));
          one *= ((xi - x[j] > 0) ? 1 : -1);
        }
      }
      w[i] = one /
             mpfr::exp(denom + mpfr::log(mpfr::mpreal(2.0)) * (x.size() - 1));
    }
  } else {
    std::size_t step = (x.size() - 2) / 15 + 1;
    mpreal one = 1u;
    for (std::size_t i = 0u; i < x.size(); ++i) {
      mpreal denom = 1.0;
      mpreal xi = x[i];
      for (std::size_t j = 0u; j < step; ++j) {
        for (std::size_t k = j; k < x.size(); k += step)
          if (k != i)
            denom *= ((xi - x[k]) << 1);
      }
      w[i] = one / denom;
    }
  }

  mpreal::set_default_prec(prevPrec);
}

void barycentricWeights(std::vector<mpfr::mpreal> &w,
                        std::vector<mpfr::mpreal> &x,
                        std::vector<mpfr::mpreal> &fx,
                        std::pair<std::size_t, std::size_t> &deg,
                        mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  if (deg.second > 0u) {
    MatrixXm A(deg.first + deg.second, deg.first + deg.second + 1u);
    for (std::size_t j = 0u; j < deg.first + deg.second + 1u; ++j)
      A(0, j) = 1;
    for (std::size_t j = 0u; j < deg.first + deg.second + 1u; ++j)
      for (std::size_t i = 1u; i < deg.first; ++i)
        A(i, j) = A(i - 1, j) * x[j];
    for (std::size_t j = 0u; j < deg.first + deg.second + 1u; ++j)
      A(deg.first, j) = fx[j];
    for (std::size_t j = 0u; j < deg.first + deg.second + 1u; ++j)
      for (std::size_t i = deg.first + 1u; i < deg.first + deg.second; ++i)
        A(i, j) = A(i - 1, j) * x[j];

    MatrixXm ker = A.fullPivLu().kernel();
    if (ker.cols() != 1u) {
      std::cerr << "BARY: Nullspace has rank " << ker.cols() << std::endl;
      exit(EXIT_FAILURE);
    }
    w.resize(deg.first + deg.second + 1u);
    for (std::size_t i{0u}; i < w.size(); ++i)
      w[i] = ker(i);
  } else {
    // std::cout << "Polynomial!\n";
    barycentricWeights(w, x, prec);
  }
  mpreal::set_default_prec(prevPrec);
}

void barycentricWeights(std::vector<double> &w, std::vector<double> &x,
                        std::vector<double> &fx,
                        std::pair<std::size_t, std::size_t> &deg) {

  MatrixXd A(deg.first + deg.second, deg.first + deg.second + 1u);
  for (std::size_t j = 0u; j < deg.first + deg.second + 1u; ++j)
    A(0, j) = 1;
  for (std::size_t j = 0u; j < deg.first + deg.second + 1u; ++j)
    for (std::size_t i = 1u; i < deg.first; ++i)
      A(i, j) = A(i - 1, j) * x[j];
  for (std::size_t j = 0u; j < deg.first + deg.second + 1u; ++j)
    A(deg.first, j) = fx[j];
  for (std::size_t j = 0u; j < deg.first + deg.second + 1u; ++j)
    for (std::size_t i = deg.first + 1u; i < deg.first + deg.second; ++i)
      A(i, j) = A(i - 1, j) * x[j];

  MatrixXd ker = A.fullPivLu().kernel();
  if (ker.cols() != 1u) {
    std::cerr << "BARY: Nullspace has rank " << ker.cols() << std::endl;
    std::cout << ker << std::endl;
    exit(EXIT_FAILURE);
  }
  w.resize(deg.first + deg.second + 1u);
  for (std::size_t i{0u}; i < w.size(); ++i)
    w[i] = ker(i);
}

void evaluateBary(mpfr::mpreal &result, const mpfr::mpreal &xVal,
                  std::vector<mpfr::mpreal> &x, std::vector<mpfr::mpreal> &fx,
                  std::vector<mpfr::mpreal> &w, mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  mpfr::mpreal num, denom;
  mpfr::mpreal buff;
  num = denom = 0;

  result = xVal;
  std::size_t r = x.size();
  for (std::size_t i = 0u; i < r; ++i) {
    if (result == x[i]) {
      result = fx[i];
      mpreal::set_default_prec(prevPrec);
      return;
    }
    buff = w[i] / (xVal - x[i]);
    num = fma(buff, fx[i], num);
    denom += buff;
  }
  result = num / denom;
  mpreal::set_default_prec(prevPrec);
}

void evaluateBary(double &result, double &xVal, std::vector<double> &x,
                  std::vector<double> &fx, std::vector<double> &w) {
  double num, denom;
  double buff;
  num = denom = 0;

  result = xVal;
  std::size_t r = x.size();
  for (std::size_t i = 0u; i < r; ++i) {
    if (result == x[i]) {
      result = fx[i];
      return;
    }
    buff = w[i] / (xVal - x[i]);
    num += buff * fx[i];
    denom += buff;
  }
  result = num / denom;
}
