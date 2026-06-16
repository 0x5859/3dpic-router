#include "sinic/geometry.hpp"

#include <algorithm>

namespace sinic::geo {

double cross_product(const Point& a, const Point& b, const Point& c) noexcept {
    return (b.first - a.first) * (c.second - a.second)
         - (b.second - a.second) * (c.first - a.first);
}

bool is_point_on_segment(const Point& seg_start,
                         const Point& seg_end,
                         const Point& point) noexcept {
    if (cross_product(seg_start, seg_end, point) != 0.0) {
        return false;
    }
    const bool x_in = (point.first  >= std::min(seg_start.first,  seg_end.first))
                   && (point.first  <= std::max(seg_start.first,  seg_end.first));
    const bool y_in = (point.second >= std::min(seg_start.second, seg_end.second))
                   && (point.second <= std::max(seg_start.second, seg_end.second));
    return x_in && y_in;
}

bool segments_intersect(const Point& a, const Point& b,
                        const Point& c, const Point& d) noexcept {
    // Bounding-box quick reject — keeps the cubic-in-E worst case
    // tolerable for the geometric fallback path.
    if (std::max(a.first,  b.first)  < std::min(c.first,  d.first)
     || std::max(c.first,  d.first)  < std::min(a.first,  b.first)
     || std::max(a.second, b.second) < std::min(c.second, d.second)
     || std::max(c.second, d.second) < std::min(a.second, b.second)) {
        return false;
    }

    const double c1 = cross_product(a, b, c);
    const double c2 = cross_product(a, b, d);
    const double c3 = cross_product(c, d, a);
    const double c4 = cross_product(c, d, b);

    if ((c1 * c2) < 0.0 && (c3 * c4) < 0.0) return true;

    if (c1 == 0.0 && is_point_on_segment(a, b, c)) return true;
    if (c2 == 0.0 && is_point_on_segment(a, b, d)) return true;
    if (c3 == 0.0 && is_point_on_segment(c, d, a)) return true;
    if (c4 == 0.0 && is_point_on_segment(c, d, b)) return true;

    return false;
}

static bool has_shared_endpoint(const Point& p1, const Point& p2,
                                const Point& p3, const Point& p4) noexcept {
    return (p1 == p3) || (p1 == p4) || (p2 == p3) || (p2 == p4);
}

static bool both_on_segment(const Point& seg_start, const Point& seg_end,
                            const Point& pt1, const Point& pt2) noexcept {
    return is_point_on_segment(seg_start, seg_end, pt1)
        && is_point_on_segment(seg_start, seg_end, pt2);
}

bool segments_intersect_strict(const Point& a, const Point& b,
                               const Point& c, const Point& d) noexcept {
    if (has_shared_endpoint(a, b, c, d)) return false;
    if (both_on_segment(a, b, c, d))      return false;
    if (both_on_segment(c, d, a, b))      return false;
    return segments_intersect(a, b, c, d);
}

} // namespace sinic::geo
