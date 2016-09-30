#include "minimax/cheby.h"
#include "minimax/minimax.h"
#include "minimax/plotting.h"
#include "minimax/ratbary.h"
#include "gtest/gtest.h"
#include <chrono>
#include <fstream>
#include <vector>


TEST(ratremez_tests, roots) {

    mp_prec_t prec = 500u;
    mpfr::mpreal::set_default_prec(prec);
    std::pair<std::size_t, std::size_t> degrees = std::make_pair(5u, 5u);
    std::vector<mpfr::mpreal> w;


    // Example: approximate the function x ----> sqrt((x+1)/2)
    // over [-1,1] using a degree (5, 5) rational function
    std::function<mpfr::mpreal(mpfr::mpreal)> f =
          [&](mpfr::mpreal input) -> mpfr::mpreal {

         return mpfr::sqrt((input + 1) / 2);
    };


    mpfr::mpreal delta;
    std::vector<mpfr::mpreal> x(degrees.first + degrees.second + 2u);
    generateEquidistantNodes(x, degrees.first + degrees.second + 1u, prec);
    applyCos(x, x);
    std::sort(begin(x), end(x));
    exchange(delta, x, f, degrees, 32, prec);

}
