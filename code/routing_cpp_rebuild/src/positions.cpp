#include "sinic/positions.hpp"

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace sinic {

PositionMap distribute_nodes_around_square(int nodes_per_side, double side_length) {
    PositionMap positions;
    positions.reserve(static_cast<std::size_t>(nodes_per_side) * 4);

    int idx = 0;
    const double spacing = side_length / (nodes_per_side + 1);

    // Top edge: left → right
    for (int i = 1; i <= nodes_per_side; ++i) {
        positions[idx++] = std::make_pair(spacing * i, side_length);
    }
    // Right edge: top → bottom
    for (int i = 1; i <= nodes_per_side; ++i) {
        positions[idx++] = std::make_pair(side_length, side_length - spacing * i);
    }
    // Bottom edge: right → left
    for (int i = 1; i <= nodes_per_side; ++i) {
        positions[idx++] = std::make_pair(side_length - spacing * i, 0.0);
    }
    // Left edge: bottom → top
    for (int i = 1; i <= nodes_per_side; ++i) {
        positions[idx++] = std::make_pair(0.0, spacing * i);
    }

    return positions;
}

PositionMap distribute_nodes_around_rectangle(int nodes_on_length,
                                              int nodes_on_width,
                                              double length,
                                              double width) {
    PositionMap positions;
    positions.reserve(static_cast<std::size_t>(nodes_on_length) * 2
                      + static_cast<std::size_t>(nodes_on_width) * 2);

    int idx = 0;
    const double spacing_length = length / (nodes_on_length + 1);
    const double spacing_width  = width  / (nodes_on_width  + 1);

    // Top edge (long side): left → right
    for (int i = 1; i <= nodes_on_length; ++i) {
        positions[idx++] = std::make_pair(spacing_length * i, width);
    }
    // Right edge (short side): top → bottom
    for (int i = 1; i <= nodes_on_width; ++i) {
        positions[idx++] = std::make_pair(length, width - spacing_width * i);
    }
    // Bottom edge (long side): right → left
    for (int i = 1; i <= nodes_on_length; ++i) {
        positions[idx++] = std::make_pair(length - spacing_length * i, 0.0);
    }
    // Left edge (short side): bottom → top
    for (int i = 1; i <= nodes_on_width; ++i) {
        positions[idx++] = std::make_pair(0.0, spacing_width * i);
    }

    return positions;
}

PositionMap load_positions_from_json(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) {
        throw std::runtime_error(
            "load_positions_from_json: cannot open " + path);
    }
    nlohmann::json j;
    try {
        ifs >> j;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error(
            "load_positions_from_json: malformed JSON in " + path + ": "
            + e.what());
    }
    if (!j.is_object()) {
        throw std::runtime_error(
            "load_positions_from_json: top-level must be an object "
            "{\"<node>\": [x, y], ...} in " + path);
    }
    if (j.empty()) {
        throw std::runtime_error(
            "load_positions_from_json: empty positions object in " + path);
    }
    PositionMap positions;
    positions.reserve(j.size());
    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& key = it.key();
        const bool canonical =
            !key.empty()
            && (key == "0"
                || (key[0] >= '1' && key[0] <= '9'
                    && std::all_of(key.begin() + 1, key.end(),
                                   [](unsigned char c) {
                                       return c >= '0' && c <= '9';
                                   })));
        if (!canonical) {
            throw std::runtime_error(
                "load_positions_from_json: node key '" + key
                + "' must be a canonical non-negative integer "
                  "(\"0\" or [1-9][0-9]*; no sign/space/leading zeros) in "
                + path);
        }
        int node = 0;
        try {
            node = std::stoi(key);
        } catch (const std::exception&) {
            throw std::runtime_error(
                "load_positions_from_json: node key '" + key
                + "' out of int range in " + path);
        }
        const auto& v = it.value();
        if (!v.is_array() || v.size() != 2 || !v[0].is_number()
            || !v[1].is_number()) {
            throw std::runtime_error(
                "load_positions_from_json: node " + it.key()
                + " value must be a 2-number array [x, y] in " + path);
        }
        positions[node] =
            std::make_pair(v[0].get<double>(), v[1].get<double>());
    }
    const int k = static_cast<int>(positions.size());
    for (int n = 0; n < k; ++n) {
        if (positions.find(n) == positions.end()) {
            throw std::runtime_error(
                "load_positions_from_json: keys must be the contiguous "
                "set 0.." + std::to_string(k - 1) + " (missing "
                + std::to_string(n) + ") in " + path);
        }
    }
    return positions;
}

} // namespace sinic
