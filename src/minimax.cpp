#include "minimax/minimax.h"
#include "minimax/eigenvalue.h"
#include "minimax/plotting.h"
#include "minimax/ratbary.h"

template <typename Type> void remove_duplicate(std::vector<Type> &vec) {
  std::set<Type> s(vec.begin(), vec.end());
  vec.assign(s.begin(), s.end());
}

void splitDomain(std::vector<Interval> &subIntervals,
                 std::vector<mpfr::mpreal> &x, mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  std::vector<mpfr::mpreal> bufferX = x;
  std::sort(begin(bufferX), end(bufferX));

  remove_duplicate(bufferX);
  std::sort(begin(bufferX), end(bufferX));


  if (bufferX[0u] != -1)
    subIntervals.push_back(std::make_pair(mpfr::mpreal(-1.0), bufferX[0u]));
  for (std::size_t i{1u}; i < bufferX.size(); ++i)
    subIntervals.push_back(std::make_pair(bufferX[i - 1u], bufferX[i]));
  if (bufferX[bufferX.size() - 1u] != 1)
    subIntervals.push_back(
        std::make_pair(bufferX[bufferX.size() - 1u], mpfr::mpreal(1.0)));

  mpreal::set_default_prec(prevPrec);
}

void trialApproximant(mpfr::mpreal &h, std::vector<mpfr::mpreal> &fx,
                      std::vector<mpfr::mpreal> &x,
                      std::pair<std::size_t, std::size_t> &degree,
                      mp_prec_t prec) {
  using mpfr::mpreal;
  mp_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  std::size_t N = degree.first + degree.second;
  MatrixXm Vm(N + 2u, N + 2u);

  for (std::size_t i{0u}; i < N + 2u; ++i)
    Vm(i, 0) = 1.0;

  for (std::size_t i{0u}; i < N + 2u; ++i)
    for (std::size_t j{1u}; j < N + 2u; ++j)
      Vm(i, j) = Vm(i, j - 1u) * x[i];

  Eigen::HouseholderQR<MatrixXm> qr(Vm);
  MatrixXm C = qr.householderQ();

  MatrixXm ZL = C.block(0u, degree.first + 1u, N + 2u, degree.second + 1u);
  ZL.transposeInPlace();
  for (std::size_t i{0u}; i < (std::size_t)ZL.rows(); ++i)
    for (std::size_t j{0u}; j < (std::size_t)ZL.cols(); ++j)
      ZL(i, j) *= fx[j];
  ZL = ZL * C.block(0, 0, N + 2u, degree.second + 1u);

  std::vector<mpfr::mpreal> sigma(fx.size());
  for (std::size_t i{0u}; i < sigma.size(); i += 2u)
    sigma[i] = 1;
  for (std::size_t i{1u}; i < sigma.size(); i += 2u)
    sigma[i] = -1;

  MatrixXm ZR = C.block(0u, degree.first + 1u, N + 2u, degree.second + 1u);
  ZR.transposeInPlace();
  for (std::size_t i{0u}; i < (std::size_t)ZR.rows(); ++i)
    for (std::size_t j{0u}; j < (std::size_t)ZR.cols(); ++j)
      ZR(i, j) *= sigma[j];
  ZR = ZR * C.block(0, 0, N + 2u, degree.second + 1u);

  Eigen::GeneralizedEigenSolver<MatrixXm> ges(ZL, ZR, true);
  MatrixXm d = ges.eigenvalues().real();
  MatrixXm zero(ZL.rows(), ZL.cols());
  for (std::size_t i{0u}; i < (std::size_t)ZL.rows(); ++i)
    for (std::size_t j{0u}; j < (std::size_t)ZL.cols(); ++j)
      zero(i, j) = 0.0;
  MatrixXm v(ZL.rows(), ZL.rows());
  for (std::size_t i = 0u; i < (std::size_t)v.cols(); ++i) {
    MatrixXm ker = (ZL - (d(i, 0u) * ZR)).fullPivLu().kernel();
    if (ker.cols() != 1u) {
      std::cerr << "GES: Nullspace has rank " << ker.cols() << std::endl;
      std::cerr << ker << std::endl;
      exit(EXIT_FAILURE);
    }
    v.col(i) = ker;
  }

  MatrixXm qAll = C.block(0u, 0u, N + 2u, degree.second + 1u) * v;

  bool valid{false};
  std::size_t validIndex{0u};
  for (std::size_t i{0u}; i < (std::size_t)qAll.cols(); ++i) {
    int signSum{0};
    for (std::size_t j{0u}; j < (std::size_t)qAll.rows(); ++j)
      signSum += mpfr::sgn(qAll(j, i));
    if ((std::size_t)(abs(signSum)) == (std::size_t)qAll.rows()) {
      if (!valid) {
        valid = true;
        validIndex = i;
      } else {
        // FIXME: this is again a heuristic, we need some
        // way of retrieving the real denominator which
        // doesn't change sign on the approximation domain
        std::cerr << "More than one candidate!" << std::endl;
        if (mpfr::abs(d(i)) < mpfr::abs(d(validIndex)))
          validIndex = i;
        // exit(EXIT_FAILURE);
      }
    }
  }
  if (!valid) {
    std::cerr << "No candidate!" << std::endl;
    // FIXME: This is a heuristic and it is here that we should
    // maybe modify the previous reference set
    std::size_t closestMatch{0u};
    int matchSum{0};
    int currSum{0};
    for (std::size_t i{0u}; i < (std::size_t)qAll.rows(); ++i)
      currSum += mpfr::sgn(qAll(i, closestMatch));

    matchSum = abs(currSum);

    for (std::size_t j{1u}; j < (std::size_t)qAll.cols(); ++j) {
      currSum = 0;
      for (std::size_t i{0u}; i < (std::size_t)qAll.rows(); ++i)
        currSum += mpfr::sgn(qAll(i, j));

      if (abs(currSum) > matchSum)
        closestMatch = j;
    }
    validIndex = closestMatch;
    // exit(EXIT_FAILURE);
  }

  h = d(validIndex);
  std::cout << "Levelled error: " << h << std::endl;

  mpreal::set_default_prec(prevPrec);
}

void findExtrema(mpfr::mpreal& minimaxError, mpfr::mpreal &convergenceOrder,
                 std::vector<mpfr::mpreal> &newRef,
                 std::function<mpfr::mpreal(mpfr::mpreal)> &f,
                 std::vector<mpfr::mpreal> &oldRef,
                 std::pair<std::size_t, std::size_t> &degree, int Nmax,
                 mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  // 1. Split the approximation domain
  std::vector<Interval> subIntervals;
  mpfr::mpreal a = -1;
  mpfr::mpreal b = 1;
  splitDomain(subIntervals, oldRef, prec);


  // 2. Compute the levelled error and the current
  // rational interpolant (if possible)
  mpfr::mpreal h;
  std::vector<mpfr::mpreal> oldFx(oldRef.size());
  for (std::size_t i{0u}; i < oldFx.size(); ++i) {
    oldFx[i] = f(oldRef[i]);
  }

  trialApproximant(h, oldFx, oldRef, degree, prec);
  std::vector<mpfr::mpreal> interpF(oldRef.size());
  for (std::size_t i{0u}; i < interpF.size(); i += 2u)
    interpF[i] = oldFx[i] - h;
  for (std::size_t i{1u}; i < interpF.size(); i += 2u)
    interpF[i] = oldFx[i] + h;

  std::vector<mpfr::mpreal> wx{oldRef};
  std::vector<mpfr::mpreal> wfx{interpF};
  std::size_t wSize = oldFx.size() - 1u;

  mpfr::mpreal buffer = wx[wSize];
  wx[wSize] = wx[wSize - 1u];
  wx[wSize - 1u] = buffer;
  wx.resize(wSize);

  buffer = wfx[wSize];
  wfx[wSize] = wfx[wSize - 1u];
  wfx[wSize - 1u] = buffer;
  wfx.resize(wSize);

  // interpolation zeros
  std::vector<mpfr::mpreal> xz;
  // interpolation non-zeros
  std::vector<mpfr::mpreal> xnz;
  // values at interpolation non-zeros
  std::vector<mpfr::mpreal> fxnz;
  for (std::size_t i{0u}; i < wfx.size(); ++i) {
    if (wfx[i] == 0)
      xz.push_back(wx[i]);
    else {
      xnz.push_back(wx[i]);
      fxnz.push_back(wfx[i]);
    }
  }

  std::vector<mpfr::mpreal> w;
  std::function<mpfr::mpreal(mpfr::mpreal)> funcHandle;
  std::function<mpfr::mpreal(mpfr::mpreal)> errHandle;
  std::size_t mu = xz.size();
  std::pair<std::size_t, std::size_t> nDegree{degree};

  if (degree.first - mu >= degree.second) {
    nDegree.first = degree.first - mu;
    w.resize(xnz.size());
    barycentricWeights(w, xnz, fxnz, nDegree, prec);
    funcHandle = [&](mpfr::mpreal val) -> mpfr::mpreal {
      mpfr::mpreal res;
      evaluateBary(res, val, xnz, fxnz, w, prec);
      for (std::size_t j{0u}; j < xz.size(); ++j)
        res *= (res - xz[j]);
      return res;
    };

    errHandle = [&](mpfr::mpreal val) -> mpfr::mpreal {
      mpfr::mpreal res;
      evaluateBary(res, val, xnz, fxnz, w, prec);
      for (std::size_t j{0u}; j < xz.size(); ++j)
        res *= (res - xz[j]);
      return f(val) - res;
    };

  } else {
    nDegree.first = degree.second;
    nDegree.second = degree.first - mu;
    for (std::size_t i{0u}; i < fxnz.size(); ++i)
      fxnz[i] = mpfr::mpreal(1.0) / fxnz[i];
    w.resize(xnz.size());
    barycentricWeights(w, xnz, fxnz, nDegree, prec);
    funcHandle = [&](mpfr::mpreal val) -> mpfr::mpreal {
      mpfr::mpreal res;
      evaluateBary(res, val, xnz, fxnz, w, prec);
      res = mpfr::mpreal(1.0) / res;
      for (std::size_t j{0u}; j < xz.size(); ++j)
        res *= (res - xz[j]);
      return res;
    };

    errHandle = [&](mpfr::mpreal val) -> mpfr::mpreal {
      mpfr::mpreal res;
      evaluateBary(res, val, xnz, fxnz, w, prec);
      res = mpfr::mpreal(1.0) / res;
      for (std::size_t j{0u}; j < xz.size(); ++j)
        res *= (res - xz[j]);
      return f(val) - res;
    };
  }


  // 3. Find the set of potential extrema inside each subinterval
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> potentialExtrema;

  std::vector<mpfr::mpreal> chebyNodes(Nmax + 1);
  generateEquidistantNodes(chebyNodes, Nmax, prec);
  applyCos(chebyNodes, chebyNodes);

  mpfr::mpreal extremaErrorValueLeft;
  mpfr::mpreal extremaErrorValueRight;
  mpfr::mpreal extremaErrorValue;

  std::vector<std::vector<mpfr::mpreal>> pExs(subIntervals.size());

#pragma omp parallel for
  for (std::size_t i = 0u; i < subIntervals.size(); ++i) {
    // find the Chebyshev nodes scaled to the current subinterval
    std::vector<mpfr::mpreal> siCN(Nmax + 1u);
    changeOfVariable(siCN, chebyNodes, subIntervals[i].first,
                     subIntervals[i].second);

    // compute the Chebyshev interpolation function values on the
    // current subinterval
    std::vector<mpfr::mpreal> fx(Nmax + 1u);
    for (std::size_t j = 0u; j < fx.size(); ++j) {
      fx[j] = errHandle(siCN[j]);
    }

    // compute the values of the CI coefficients and those of its
    // derivative
    std::vector<mpfr::mpreal> chebyCoeffs(Nmax + 1u);
    generateChebyshevCoefficients(chebyCoeffs, fx, Nmax, prec);
    std::vector<mpfr::mpreal> derivCoeffs(Nmax);
    derivativeCoefficients2ndKind(derivCoeffs, chebyCoeffs);

    // solve the corresponding eigenvalue problem and determine the
    // local extrema situated in the current subinterval
    MatrixXm Cm(Nmax - 1u, Nmax - 1u);
    generateColleagueMatrix2ndKind(Cm, derivCoeffs, true, prec);

    std::vector<mpfr::mpreal> eigenRoots;
    VectorXcm roots;
    determineEigenvalues(roots, Cm);
    getRealValues(eigenRoots, roots, a, b);
    changeOfVariable(eigenRoots, eigenRoots, subIntervals[i].first,
                     subIntervals[i].second);
    for (std::size_t j = 0u; j < eigenRoots.size(); ++j)
      pExs[i].push_back(eigenRoots[j]);
    pExs[i].push_back(subIntervals[i].first);
    pExs[i].push_back(subIntervals[i].second);
    /*std::stringstream subIntName;
    subIntName << "subinterval" << i;
    std::string name = subIntName.str();
    std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> points(pExs[i].size());
    for(std::size_t k{0u}; k < points.size(); ++k)
    {
            points[k].first = pExs[i][k];
            points[k].second = errHandle(pExs[i][k]);
    }
    plotFuncEtVals(name, errHandle, points,
                    subIntervals[i].first, subIntervals[i].second, prec);*/
  }

  for (std::size_t i = 0u; i < pExs.size(); ++i)
    for (std::size_t j = 0u; j < pExs[i].size(); ++j)
      potentialExtrema.push_back(
          std::make_pair(pExs[i][j], errHandle(pExs[i][j])));

  std::sort(potentialExtrema.begin(), potentialExtrema.end(),
            [](const std::pair<mpfr::mpreal, mpfr::mpreal> &lhs,
               const std::pair<mpfr::mpreal, mpfr::mpreal> &rhs) {
              return lhs.first < rhs.first;
            });

  newRef.clear();
  std::size_t extremaIt = 0u;
  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> alternatingExtrema;
  mpfr::mpreal minError = INT_MAX;
  mpfr::mpreal maxError = INT_MIN;
  mpfr::mpreal absError;

  while (extremaIt < potentialExtrema.size()) {
    std::pair<mpfr::mpreal, mpfr::mpreal> maxErrorPoint;
    maxErrorPoint = potentialExtrema[extremaIt];
    while (extremaIt < potentialExtrema.size() - 1 &&
           mpfr::sgn(maxErrorPoint.second) *
                   mpfr::sgn(potentialExtrema[extremaIt + 1].second) >
               0) {
      ++extremaIt;
      if (mpfr::abs(maxErrorPoint.second) <
          mpfr::abs(potentialExtrema[extremaIt].second))
        maxErrorPoint = potentialExtrema[extremaIt];
    }
    alternatingExtrema.push_back(maxErrorPoint);
    ++extremaIt;
  }

  std::vector<std::pair<mpfr::mpreal, mpfr::mpreal>> bufferExtrema;
  std::cout << "Alternating extrema: " << alternatingExtrema.size() << "/"
            << oldRef.size() << std::endl;

  if (alternatingExtrema.size() < oldRef.size()) {
    std::cerr << "TRIGGER: Not enough alternating extrema!\n"
              << "POSSIBLE CAUSE: Nmax too small\n";
    std::cerr << "Size: " << alternatingExtrema.size() << std::endl;
    convergenceOrder = 2.0;
    mpreal::set_default_prec(prevPrec);
    return;
  } else if (alternatingExtrema.size() > oldRef.size()) {
    std::size_t remSuperfluous = alternatingExtrema.size() - oldRef.size();
    if (remSuperfluous % 2 != 0) {
      mpfr::mpreal abs1 = mpfr::abs(alternatingExtrema[0].second);
      mpfr::mpreal abs2 =
          mpfr::abs(alternatingExtrema[alternatingExtrema.size() - 1].second);
      std::size_t sIndex = 0u;
      if (abs1 < abs2)
        sIndex = 1u;
      for (std::size_t i = sIndex; i < alternatingExtrema.size() + sIndex - 1u;
           ++i)
        bufferExtrema.push_back(alternatingExtrema[i]);
      alternatingExtrema = bufferExtrema;
      bufferExtrema.clear();
    }

    while (alternatingExtrema.size() > oldRef.size()) {
      std::size_t toRemoveIndex = 0u;
      mpfr::mpreal minValToRemove =
          mpfr::max(mpfr::abs(alternatingExtrema[0].second),
                    mpfr::abs(alternatingExtrema[1].second));
      mpfr::mpreal removeBuffer;
      for (std::size_t i = 1u; i < alternatingExtrema.size() - 1; ++i) {
        removeBuffer = mpfr::max(mpfr::abs(alternatingExtrema[i].second),
                                 mpfr::abs(alternatingExtrema[i + 1].second));
        if (removeBuffer < minValToRemove) {
          minValToRemove = removeBuffer;
          toRemoveIndex = i;
        }
      }
      for (std::size_t i = 0u; i < toRemoveIndex; ++i)
        bufferExtrema.push_back(alternatingExtrema[i]);
      for (std::size_t i = toRemoveIndex + 2u; i < alternatingExtrema.size();
           ++i)
        bufferExtrema.push_back(alternatingExtrema[i]);
      alternatingExtrema = bufferExtrema;
      bufferExtrema.clear();
    }
  }
  if (alternatingExtrema.size() < oldRef.size()) {
    std::cerr << "Trouble!\n";
    exit(EXIT_FAILURE);
  }

  std::string testFile2 = "output2";

  plotFuncEtVals(testFile2, errHandle, alternatingExtrema, a, b, prec);

  newRef.clear();
  for (auto &it : alternatingExtrema) {
    newRef.push_back(it.first);
    absError = mpfr::abs(it.second);
    minError = mpfr::min(minError, absError);
    maxError = mpfr::max(maxError, absError);
  }

  std::cout << "Min error = " << minError << std::endl;
  std::cout << "Max error = " << maxError << std::endl;
  convergenceOrder = (maxError - minError) / maxError;
  std::cout << "Convergence order = " << convergenceOrder << std::endl;
  minimaxError = minError;

  mpreal::set_default_prec(prevPrec);
}

void exchange(mpfr::mpreal& delta, std::vector<mpfr::mpreal> &x,
              std::function<mpfr::mpreal(mpfr::mpreal)> &f,
              std::pair<std::size_t, std::size_t> &degree, int Nmax,
              mp_prec_t prec) {
  using mpfr::mpreal;
  mpfr_prec_t prevPrec = mpreal::get_default_prec();
  mpreal::set_default_prec(prec);

  mpfr::mpreal convergenceOrder = 1;
  std::vector<mpfr::mpreal> newX;
  std::size_t iterationCount = 1u;
  do {
    std::cout << "===============ITERATION " << iterationCount
              << "==============" << std::endl;
    findExtrema(delta, convergenceOrder, newX, f, x, degree, Nmax, prec);
    x = newX;
    //for (std::size_t i{1u}; i < x.size(); ++i)
      //std::cout << (x[i] - x[i - 1]).toString("%.30RNe") << ";\n";
    // std::cout << x[i].toString("%.30RNe") << ";\n";
    std::cout << std::endl;
    ++iterationCount;
  } while (convergenceOrder > 0.1e-20);
  x = newX;
  mpreal::set_default_prec(prevPrec);
}
