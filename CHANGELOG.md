# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.2.0] - 2026-08-13

A third protocol backend, hand-rolled from scratch against the wire format —
no runtime dependency on `ion-c` or `ion-rust`.

### Added

- **Ion binary** (`zerialize::Ion`, [`protocols/ion.hpp`](include/zerialize/protocols/ion.hpp)):
  both the reader and writer are hand-rolled. `ion-c`'s reader is a single
  stateful forward-only cursor with no zero-copy view type, which rules it
  out for a `Deserializer` whose `operator[]` must be `const` and safe to
  call multiple times in one expression (ordinary generic code relies on
  this; C++ doesn't guarantee argument evaluation order). The architecture
  used instead — a small, copyable value handle plus a `shared_ptr`-shared,
  once-resolved symbol table context — mirrors `ion-rust`'s `lazy` module
  (`LazyValue`/`LazyStruct`, `Copy`, "immutable; fields can be read any
  number of times") and reuses the same lifetime pattern `beve.hpp` already
  uses for the same reason (a raw pointer back to a document that could
  move on copy is exactly the class of bug that produced a real
  heap-buffer-overflow in glaze's own lazy reader). Every wire-format
  detail (VarUInt's continuation-bit convention — the opposite of
  protobuf/LEB128's; big-endian magnitude/float encoding — the opposite of
  BEVE's little-endian; the exact local symbol table struct shape) was
  verified against real bytes produced by `ion-c`'s own writer, and the
  reader was additionally cross-validated by parsing that real `ion-c`
  output independently of our own writer. Unlike BSON, Ion values are
  self-describing at the root, so there's no root document/array ambiguity
  and no bespoke test carve-out is needed. Enabled via `ZERIALIZE_ENABLE_ION`
  (default **on** — no new dependency, no C++ standard change). See
  [`protocols/Ion.md`](include/zerialize/protocols/Ion.md) for the wire
  format, symbol table design, and known v1 limitations (decimal/timestamp/
  clob are skippable but not yet exposed as scalars; shared/imported symbol
  tables are rejected with a clear error rather than silently misread).

## [1.1.0] - 2026-08-13

Two new protocol backends.

### Added

- **BEVE** (`zerialize::Beve`, [`protocols/beve.hpp`](include/zerialize/protocols/beve.hpp)):
  the `Deserializer` adapts [glaze](https://github.com/stephenberry/glaze)'s
  `lazy_beve_document`/`lazy_beve_view` for zero-copy, bounds-checked reading;
  the `Serializer` is a hand-rolled writer with no runtime dependency on
  glaze (glaze's own write path can't produce BEVE's typed-array encoding
  from a dynamic value tree, which would silently break blob
  round-tripping). Opt-in via `ZERIALIZE_ENABLE_BEVE` (default **off**),
  since glaze's own `CMakeLists.txt` hard-requires C++23, unlike every
  other protocol here, which build under this project's normal C++20
  baseline. See [`protocols/BEVE.md`](include/zerialize/protocols/BEVE.md).
- **BSON** (`zerialize::Bson`, [`protocols/bson.hpp`](include/zerialize/protocols/bson.hpp)):
  the `Serializer` wraps `jsoncons::bson::bson_bytes_encoder` (already a
  dependency, for CBOR); the `Deserializer` is hand-rolled and independent
  of jsoncons, matching the precedent `cbor.hpp` already set. Enabled via
  `ZERIALIZE_ENABLE_BSON` (default **on** — no new dependency, no C++
  standard change). Documents BSON's two format-inherent constraints (no
  top-level scalar; no unsigned 64-bit wire type) in
  [`protocols/BSON.md`](include/zerialize/protocols/BSON.md).

### Security

- **BEVE**: found and fixed a heap-buffer-overflow reachable from untrusted
  input in the underlying library: glaze v8.0.0's lazy
  `operator[](size_t)` on generic arrays doesn't verify the returned
  element pointer is within the buffer when the declared element count
  exceeds what's actually present. `BeveDeserializer` now bounds-checks
  every indexing/lookup result itself before trusting it (confirmed fixed
  under AddressSanitizer).

## [1.0.1] - 2026-08-02

A security/correctness hardening pass across every protocol backend, found and
verified with AddressSanitizer/UndefinedBehaviorSanitizer, plus a build
reliability fix and dependency pinning.

### Security

- **CBOR**: fixed three heap-buffer-overflow reads reachable from
  malformed/truncated input:
  - `asDouble()`/`asFloat()` read the float16/32/64 payload with no bounds
    check.
  - `contains()`'s map-key fast path built a `string_view` from unchecked
    header info instead of the bounds-checked `asString()`.
  - `KeysView` (`mapKeys()`) carried its own private copy of the
    header-parsing logic that omitted nearly all of the bounds checks the
    canonical implementation has.
- **MessagePack**: fixed the same class of bug, more pervasively — the
  `str_info()`, `bin_info()`, `arr_info()`, and `map_info()` header-parsing
  helpers read multi-byte length prefixes with no bounds checking at all, and
  `mp_skip()` validated headers but not payload lengths for string/binary
  markers, letting truncated input produce an out-of-range offset passed to
  `std::span::subspan()` (undefined behavior, not a safe throw).
- **FlexBuffers**: fixed an out-of-bounds read in `string()`/`key()` — both
  passed `std::string_view::data()/size()` straight to flatbuffers APIs that
  read one byte past the end expecting a trailing NUL (a guarantee
  `std::string_view`, unlike `std::string`, doesn't provide).
- **Tensor deserialization (Eigen)**: added an overflow check to
  `asEigenMatrixView()`'s `rows * cols * sizeof(T)` byte-size validation. A
  crafted tensor shape could previously wrap the check on 64-bit `size_t` and
  pass validation with a mismatched buffer, causing an uncontrolled
  allocation / crash instead of a clean `DeserializationError`.

### Fixed

- `zera::RootSerializer::finish()` and four related call sites (`zera`'s
  `contains()`/`operator[]`, and several tensor-deserialization `memcpy`s in
  `eigen.hpp`/`xtensor.hpp`) called `memcpy`/`memcmp` with a possibly-null
  source pointer from an empty container (e.g. a zero-element tensor, a
  0-row/0-column matrix, or an empty-string map key) — technically undefined
  behavior even for a zero-length copy. Guarded all of them, matching the
  convention already used elsewhere in these files.
- `fixed_string::c_str()` failed to compile whenever instantiated (called
  `.data()` on a raw `char[N]` array instead of returning it directly).
- `MsgPackRootSerializer` had no copy/move control despite owning a raw
  `msgpack_sbuffer` pointer, risking a double-free if ever copied; added
  explicit copy-deletion and a correct move that re-anchors the internal
  packer state.
- `json::RootSerializer` leaked its `yyjson_mut_doc*` if a `Writer` call (or
  the source `Reader` being translated from, via `translate<JSON>()`) threw
  before `finish()` was reached — happened on every failed deserialization
  translated to JSON, not just contrived cases.
- Fixed a header-only flatbuffers build failure that only manifested on a
  genuinely fresh build (`undefined reference to
  flatbuffers::ClassicLocale::instance_`) — flatbuffers' locale-handling code
  needs a translation unit this project never compiles; forced
  `FLATBUFFERS_LOCALE_INDEPENDENT=0` so the affected headers fall back to
  plain `strtod`/`strtoll` instead.
- `zbuffer.hpp` was missing `#include <memory>` (`ManagedPtr` uses
  `std::unique_ptr`), only surfacing when building this repo standalone.

### Changed

- Pinned `flatbuffers` and `msgpack-c`, which were previously tracking
  floating `master`/`c_master` branches, to explicit release tags
  (`v25.12.19`, `c-7.0.1`) for reproducible builds.
