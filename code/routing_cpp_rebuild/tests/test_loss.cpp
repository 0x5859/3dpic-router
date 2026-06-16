#include <catch2/catch.hpp>

#include "sinic/loss.hpp"

using sinic::EdgeList;
using sinic::EdgeData;
using sinic::compute_per_edge_loss;
using sinic::apply_interlayer_loss;

TEST_CASE("compute_per_edge_loss legacy formula", "[loss]") {
    // pre-M5 formula:  loss = 2 * loss_crossing * crossings + 2 * layer * loss_taper
    EdgeData a;
    a.crossings = 3;
    a.layer     = 0;
    EdgeData b;
    b.crossings = 1;
    b.layer     = 1;
    EdgeList edges{{0, 1, a}, {1, 2, b}};

    compute_per_edge_loss(edges, /*loss_crossing=*/0.5, /*loss_taper=*/2.0);

    REQUIRE(std::get<2>(edges[0]).loss == Approx(0.5 * 3 * 2 + 2 * 0 * 2.0));
    REQUIRE(std::get<2>(edges[1]).loss == Approx(0.5 * 1 * 2 + 2 * 1 * 2.0));
}

TEST_CASE("apply_interlayer_loss additive", "[loss]") {
    EdgeData a;
    a.loss                 = 1.0;
    a.interlayerCrossings  = 4;
    EdgeList edges{{0, 1, a}};
    apply_interlayer_loss(edges, /*loss_interlayer=*/0.25);
    REQUIRE(std::get<2>(edges[0]).loss == Approx(1.0 + 4 * 0.25));
}
