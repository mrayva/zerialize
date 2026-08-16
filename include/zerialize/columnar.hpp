#pragma once

#include <zerialize/concepts.hpp>
#include <zerialize/translate.hpp>
#include <string>
#include <vector>
#include <stdexcept>

namespace zerialize {

/*
 * columnar.hpp
 * ------------
 * Generic, format-agnostic conversion between row-oriented and
 * column-oriented (columnar/SoA) shapes of the same tabular data, built on
 * the same Reader/Writer bridge translate.hpp uses (write_value()) rather
 * than an intermediate tree (dyn::Value is serialize-only and has no read
 * API - see dynamic.hpp - so it isn't a fit here anyway).
 *
 * Row shape:      [ {col1: v, col2: v, ...}, {col1: v, col2: v, ...}, ... ]
 * Columnar shape: { col1: [v, v, ...], col2: [v, v, ...], ... }
 *
 * This is deliberately strict, not best-effort: a columnar record must have
 * every field be an array, and all of those arrays must be the same
 * length (this is exactly the shape pg_zerialize's rows_to_<fmt>_columnar()
 * produces - see zerialize's own consumers). A row-array must have every
 * element be an object, and every object must have the same set of field
 * names. Anything else throws std::runtime_error with a message identifying
 * what was wrong and, where applicable, which row/column - malformed input
 * should fail loudly rather than silently produce a mismatched or
 * partially-converted result.
 *
 * Empty input is the one place both directions agree on a shape with no
 * information to recover: an empty array of rows collapses to an empty
 * object ({}), and an empty object expands to an empty array ([]) - not an
 * error, matching the same empty-input convention pg_zerialize's own
 * columnar batch encoder already established.
 *
 * Provides:
 *   - `write_expanded_columnar(v, w)` / `write_collapsed_columnar(v, w)`:
 *     the low-level Reader -> Writer primitives, mirroring write_value()'s
 *     role in translate.hpp.
 *   - `expand_columnar<DstP>(src)` / `collapse_columnar<DstP>(src)`:
 *     convenience wrappers mirroring translate<DstP>(src) - build a
 *     DstP::Deserializer directly, so a caller with a columnar-shaped
 *     MsgPack reader can get back a row-shaped MsgPack (or JSON, or any
 *     other protocol) deserializer in one call.
 */

// ==== expand: columnar map -> array of row-objects ======================
template<class V, class W>
inline void write_expanded_columnar(const V& v, W& w) {
    if (!v.isMap()) {
        throw std::runtime_error(
            "write_expanded_columnar: root value must be an object (columnar record), "
            "not an array or scalar");
    }

    // First pass: collect column names, validate every field is an array,
    // and that all arrays share the same length N. Column values aren't
    // held onto here (only re-fetched by key in the emission loop below):
    // a protocol's map "value view" type (what operator[]/mapEntries()
    // return) isn't guaranteed to be the same C++ type as V itself - e.g.
    // Flex's FlexDeserializer vs. its own FlexValue sub-view - so there's
    // no single concrete type to store a heterogeneous column list as
    // without type erasure. Using a generic (auto&&) lambda sidesteps that
    // by letting each call site's own deduced type flow through untouched.
    std::vector<std::string> keys;
    std::size_t n = 0;
    bool first_column = true;

    auto note_column = [&](std::string_view key, auto&& value) {
        if (!value.isArray()) {
            throw std::runtime_error(
                "write_expanded_columnar: field \"" + std::string(key) +
                "\" is not an array - every field of a columnar record must be "
                "an array of per-row values");
        }
        const std::size_t size = value.arraySize();
        if (first_column) {
            n = size;
            first_column = false;
        } else if (size != n) {
            throw std::runtime_error(
                "write_expanded_columnar: column \"" + std::string(key) + "\" has " +
                std::to_string(size) + " value(s), but an earlier column has " +
                std::to_string(n) +
                " - every column in a columnar record must be the same length");
        }
        keys.emplace_back(key);
    };

    if constexpr (requires { v.mapEntries(); }) {
        for (auto&& entry : v.mapEntries()) note_column(entry.key, entry.value);
    } else {
        for (std::string_view k : v.mapKeys()) note_column(k, v[k]);
    }

    w.begin_array(n);
    for (std::size_t i = 0; i < n; ++i) {
        w.begin_map(keys.size());
        for (const auto& key : keys) {
            w.key(key);
            write_value(v[key][i], w);
        }
        w.end_map();
    }
    w.end_array();
}

// ==== collapse: array of row-objects -> columnar map =====================
template<class V, class W>
inline void write_collapsed_columnar(const V& v, W& w) {
    if (!v.isArray()) {
        throw std::runtime_error(
            "write_collapsed_columnar: root value must be an array of row-objects, "
            "not an object or scalar");
    }

    const std::size_t n = v.arraySize();
    if (n == 0) {
        w.begin_map(0);
        w.end_map();
        return;
    }

    // Column order/key set comes from row 0; every other row is validated
    // against it up front (same key count, same keys present) before any
    // output is written - binary Writers generally can't roll back a
    // partially-written document, so a mismatch found mid-emission would
    // leave a corrupt result.
    auto row0 = v[std::size_t{0}];
    if (!row0.isMap()) {
        throw std::runtime_error("write_collapsed_columnar: row 0 is not an object");
    }
    std::vector<std::string> keys;
    if constexpr (requires { row0.mapEntries(); }) {
        for (auto&& entry : row0.mapEntries()) keys.emplace_back(entry.key);
    } else {
        for (std::string_view k : row0.mapKeys()) keys.emplace_back(k);
    }

    for (std::size_t i = 1; i < n; ++i) {
        auto row = v[i];
        if (!row.isMap()) {
            throw std::runtime_error(
                "write_collapsed_columnar: row " + std::to_string(i) + " is not an object");
        }
        std::size_t row_field_count = 0;
        if constexpr (requires { row.mapEntries(); }) {
            for (auto&& /*unused*/ _ : row.mapEntries()) (void)_, ++row_field_count;
        } else {
            for (auto&& /*unused*/ _ : row.mapKeys()) (void)_, ++row_field_count;
        }
        if (row_field_count != keys.size()) {
            throw std::runtime_error(
                "write_collapsed_columnar: row " + std::to_string(i) + " has " +
                std::to_string(row_field_count) + " field(s), but row 0 has " +
                std::to_string(keys.size()) +
                " - every row must have the same set of fields");
        }
        for (const auto& key : keys) {
            if (!row.contains(key)) {
                throw std::runtime_error(
                    "write_collapsed_columnar: row " + std::to_string(i) +
                    " is missing field \"" + key + "\" (present in row 0)");
            }
        }
    }

    w.begin_map(keys.size());
    for (const auto& key : keys) {
        w.key(key);
        w.begin_array(n);
        for (std::size_t i = 0; i < n; ++i) {
            write_value(v[i][key], w);
        }
        w.end_array();
    }
    w.end_map();
}

// ==== Convenience: Reader -> destination Protocol, same pattern as
// translate<DstP>() in translate.hpp. ======================================

template<class DstP, class SrcV>
requires Protocol<DstP> && Reader<SrcV>
inline typename DstP::Deserializer expand_columnar(const SrcV& src) {
    using Root   = typename DstP::RootSerializer;
    using Writer = typename DstP::Serializer;

    Root   rs{};
    Writer w{rs};
    write_expanded_columnar(src, w);

    ZBuffer out = rs.finish();
    if constexpr (requires { out.to_vector_copy(); }) {
        return typename DstP::Deserializer(out.to_vector_copy());
    } else {
        return typename DstP::Deserializer(
            reinterpret_cast<const std::uint8_t*>(out.data()), out.size());
    }
}

template<class DstP, class SrcV>
requires Protocol<DstP> && Reader<SrcV>
inline typename DstP::Deserializer collapse_columnar(const SrcV& src) {
    using Root   = typename DstP::RootSerializer;
    using Writer = typename DstP::Serializer;

    Root   rs{};
    Writer w{rs};
    write_collapsed_columnar(src, w);

    ZBuffer out = rs.finish();
    if constexpr (requires { out.to_vector_copy(); }) {
        return typename DstP::Deserializer(out.to_vector_copy());
    } else {
        return typename DstP::Deserializer(
            reinterpret_cast<const std::uint8_t*>(out.data()), out.size());
    }
}

} // namespace zerialize
