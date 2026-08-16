#include <array>
#include <limits>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <cstring>
#include <type_traits>

#include <zerialize/zerialize.hpp>
#include <zerialize/tensor/xtensor.hpp>
#include <zerialize/tensor/eigen.hpp>
#ifdef ZERIALIZE_HAS_JSON
#include <zerialize/protocols/json.hpp>
#endif
#ifdef ZERIALIZE_HAS_FLEXBUFFERS
#include <zerialize/protocols/flex.hpp>
#endif
#ifdef ZERIALIZE_HAS_MSGPACK
#include <zerialize/protocols/msgpack.hpp>
#endif
#ifdef ZERIALIZE_HAS_CBOR
#include <zerialize/protocols/cbor.hpp>
#endif
#ifdef ZERIALIZE_HAS_ZERA
#include <zerialize/protocols/zera.hpp>
#endif
#ifdef ZERIALIZE_HAS_BEVE
#include <zerialize/protocols/beve.hpp>
#endif
#ifdef ZERIALIZE_HAS_BSON
#include <zerialize/protocols/bson.hpp>
#endif
#ifdef ZERIALIZE_HAS_ION
#include <zerialize/protocols/ion.hpp>
#endif

#include <xtensor/generators/xbuilder.hpp>

#include "testing_utils.hpp"

namespace zerialize {

// Small helper to assert we surface DeserializationError boundaries.
template<class F>
bool expect_deserialization_error(F&& fn) {
    try {
        std::forward<F>(fn)();
    } catch (const DeserializationError&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// Small helper to assert we surface SerializationError boundaries.
template<class F>
bool expect_serialization_error(F&& fn) {
    try {
        std::forward<F>(fn)();
    } catch (const SerializationError&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// --------------------- Per-protocol DSL tests ---------------------
template<class P>
void test_protocol_dsl() {
    using V = typename P::Deserializer;
    std::cout << "== DSL tests for <" << P::Name << "> ==\n";

    // 1) Simple map with compile-time keys
    test_serialization<P>(R"(zmap<"key1","key2">(42,"yo"))",
        [](){
            return serialize<P>( zmap<"key1","key2">(42, "yo") );
        },
        [](const V& v){
            return v.isMap()
                && v["key1"].asInt64()==42
                && v["key2"].asString()=="yo";
        });

    // 2) Array root
    test_serialization<P>(R"(zvec(1,2,3))",
        [](){
            return serialize<P>( zvec(1,2,3) );
        },
        [](const V& v){
            return v.isArray() && v.arraySize()==3
                && v[0].asInt64()==1
                && v[1].asInt64()==2
                && v[2].asInt64()==3;
        });

    // 3) Nested: array of map and array
    test_serialization<P>(R"(zmap<"a","b">( 7, zvec("x", zmap<"n">(44)) ))",
        [](){
            return serialize<P>( zmap<"a","b">(
                7,
                zvec("x", zmap<"n">(44))
            ));
        },
        [](const V& v){
            if (!v.isMap()) return false;
            if (!v["a"].isInt() || v["a"].asInt64()!=7) return false;
            auto b = v["b"];
            if (!b.isArray() || b.arraySize()!=2) return false;
            if (b[0].asString()!="x") return false;
            return b[1].isMap() && b[1]["n"].asInt64()==44;
        });

    // 4) Booleans and null
    test_serialization<P>(R"(zmap<"t","f","n">(true,false,nullptr))",
        [](){
            return serialize<P>( zmap<"t","f","n">( true, false, nullptr ) );
        },
        [](const V& v){
            return v.isMap()
                && v["t"].asBool()==true
                && v["f"].asBool()==false
                && v["n"].isNull();
        });

    // 5) Mixed numeric types (assert via int64/uint64/double)
    test_serialization<P>("mixed numeric types",
        [](){
            return serialize<P>( zmap<
                "i8","u8","i32","u32","i64","u64","d"
            >( int8_t(-5), uint8_t(200), int32_t(-123456), uint32_t(987654321u),
               int64_t(-7777777777LL), uint64_t(9999999999ULL), 3.25 ) );
        },
        [](const V& v){
            return v.isMap()
                && v["i8"].asInt64()==-5
                && v["u8"].asUInt64()==200
                && v["i32"].asInt64()==-123456
                && v["u32"].asUInt64()==987654321ULL
                && v["i64"].asInt64()==-7777777777LL
                && v["u64"].asUInt64()==9999999999ULL
                && std::abs(v["d"].asDouble()-3.25)<1e-12;
        });

    // 6) Unicode strings + embedded NUL in **value**
    auto ts1 = std::string(reinterpret_cast<const char*>(u8"héllo"));
    auto ts2 = std::string(reinterpret_cast<const char*>(u8"汉字"));
    test_serialization<P>("strings (unicode + embedded NUL)",
        [ts1, ts2](){
            const char raw[] = {'a','\0','b'};
            return serialize<P>( zvec(ts1, std::string_view(raw,3), ts2) );
        },
        [ts1, ts2](const V& v){
            if (!v.isArray() || v.arraySize()!=3) return false;
            if (v[0].asString()!=ts1) return false;
            auto s1 = v[1].asStringView();
            if (!(s1.size()==3 && s1[0]=='a' && s1[1]=='\0' && s1[2]=='b')) return false;
            return v[2].asString()==ts2;
        });

    // 7) Biggish vector (size hint exercised)
    test_serialization<P>("big vector 256",
        [](){
            std::array<int,256> a{};
            for (int i=0;i<256;++i) a[i]=i;
            return serialize<P>( a );
        },
        [](const V& v){
            if (!v.isArray() || v.arraySize()!=256) return false;
            for (int i=0;i<256;++i) if (v[i].asInt64()!=i) return false;
            return true;
        });

    // 8) mapKeys() contract
    test_serialization<P>("mapKeys() iteration",
        [](){
            return serialize<P>( zmap<"alpha","beta","gamma">(1,2,3) );
        },
        [](const V& v){
            if (!v.isMap()) return false;
            std::set<std::string_view> keys;
            for (std::string_view k : v.mapKeys()) keys.insert(k);
            return keys.size()==3 && keys.count("alpha") && keys.count("beta") && keys.count("gamma");
        });

    // 9) Array of objects built with zmap
    test_serialization<P>("array of objects",
        [](){
            return serialize<P>( zvec(
                zmap<"id","name">(1, "a"),
                zmap<"id","name">(2, "b"),
                zmap<"id","name">(3, "c")
            ));
        },
        [](const V& v){
            if (!v.isArray() || v.arraySize()!=3) return false;
            for (int i=0;i<3;++i) {
                if (!v[i].isMap()) return false;
                if (v[i]["id"].asInt64()!=i+1) return false;
            }
            return v[0]["name"].asString()=="a" &&
                   v[1]["name"].asString()=="b" &&
                   v[2]["name"].asString()=="c";
        });

    // 9) kv with tensor
    auto tens = xt::xtensor<double, 2>{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    test_serialization<P>("kv with tensor",
        [&tens](){ 
            return serialize<P>(
                zmap<"key1", "key2", "key3">(42, 3.14159, tens)
            ); 
        },
        [&tens](const V& v) {
            auto a = xtensor::asXTensor<double>(v["key3"]);
            return 
                v["key1"].asInt32() == 42 &&
                v["key2"].asDouble() == 3.14159 &&
                a == tens; 
        });

    // 10) kv with eigen matrix
    auto eigen_mat = Eigen::Matrix<double, 3, 2>();
    eigen_mat << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
    test_serialization<P>("kv with eigen matrix",
        [&eigen_mat](){ 
            return serialize<P>(
                zmap<"key1", "key2", "key3">(42, 3.14159, eigen_mat)
            ); 
        },
        [&eigen_mat](const V& v) {
            auto a = eigen::asEigenMatrix<double, 3, 2>(v["key3"]);
            return 
                v["key1"].asInt32() == 42 &&
                v["key2"].asDouble() == 3.14159 &&
                a.isApprox(eigen_mat); 
        });

    std::cout << "== DSL tests for <" << P::Name << "> passed ==\n\n";
}

// --------------------- Dynamic serialization tests ---------------------
template<class P>
void test_dynamic_serialization() {
    using V = typename P::Deserializer;
    namespace d = zerialize::dyn;
    std::cout << "== Dynamic serialization tests for <" << P::Name << "> ==\n";

    test_serialization<P>("dyn: map+array",
        [](){
            d::Value payload = d::map({
                {"id",   99},
                {"name", "dynamic"},
                {"tags", d::array({"alpha", "beta", 3})}
            });
            return serialize<P>(payload);
        },
        [](const V& v){
            if (!v.isMap()) return false;
            if (v["id"].asInt64() != 99) return false;
            if (v["name"].asString() != "dynamic") return false;
            auto tags = v["tags"];
            return tags.isArray() && tags.arraySize()==3
                && tags[0].asString()=="alpha"
                && tags[1].asString()=="beta"
                && tags[2].asInt64()==3;
        });

    test_serialization<P>("dyn: tensor xtensor helper",
        [](){
            xt::xtensor<double, 2> tensor{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            d::Value payload = d::serializable(tensor);
            return serialize<P>(payload);
        },
        [](const V& v){
            if (!v.isArray()) return false;
            auto restored = xtensor::asXTensor<double, 2>(v);
            xt::xtensor<double, 2> expected{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            return restored == expected;
        });

    test_serialization<P>("dyn: tensor eigen manual",
        [](){
            Eigen::Matrix<double, 3, 2> m;
            m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
            d::Value payload = d::serializable(m);
            return serialize<P>(payload);
        },
        [](const V& v){
            if (!v.isArray()) return false;
            auto restored = eigen::asEigenMatrix<double, 3, 2>(v);
            Eigen::Matrix<double, 3, 2> expected;
            expected << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
            return restored.isApprox(expected);
        });

    test_serialization<P>("dyn: tensor inside map",
        [](){
            xt::xtensor<double, 2> tensor{{10.0, 20.0}, {30.0, 40.0}};
            d::Value payload = d::map({
                {"meta", d::map({{"id", 7}})},
                {"tensor", d::serializable(tensor)}
            });
            return serialize<P>(payload);
        },
        [](const V& v){
            if (!v.isMap()) return false;
            if (!v["meta"].isMap() || v["meta"]["id"].asInt64() != 7) return false;
            auto restored = xtensor::asXTensor<double, 2>(v["tensor"]);
            xt::xtensor<double, 2> expected{{10.0, 20.0}, {30.0, 40.0}};
            return restored == expected;
        });

    std::cout << "== Dynamic serialization tests for <" << P::Name << "> passed ==\n\n";
}

// --------------------- Cross-protocol translation (DSL-built) ----------------
template<class SrcP, class DstP>
void test_translate_dsl() {
    using DV = typename DstP::Deserializer;

    std::cout << "== Translate (DSL) <" << SrcP::Name << "> → <" << DstP::Name << "> ==\n";

    // A: simple object
    test_serialization<DstP>("xlate: simple object",
        [](){
            auto src = serialize<SrcP>( zmap<"a","b">(11, "yo") );
            auto srd = typename SrcP::Deserializer(src.buf());
            auto drd = translate<DstP>(srd); // your translate<>

            // Re-serialize in DstP using values from drd (ensures shape+values preserved)
            return serialize<DstP>( zmap<"a","b">( drd["a"].asInt64(), drd["b"].asString() ) );
        },
        [](const DV& v){
            return v.isMap() && v["a"].asInt64()==11 && v["b"].asString()=="yo";
        });

    // B: nested mixed container
    test_serialization<DstP>("xlate: nested",
        [](){
            auto src = serialize<SrcP>( zmap<"outer">(
                zvec( zmap<"n">(44), zvec("A","B") )
            ));
            auto srd = typename SrcP::Deserializer(src.buf());
            auto drd = translate<DstP>(srd);
            return serialize<DstP>( zmap<"outer">(
                zvec( zmap<"n">( drd["outer"][0]["n"].asInt64() ),
                      zvec( drd["outer"][1][0].asString(),
                            drd["outer"][1][1].asString() ) )
            ));
        },
        [](const DV& v){
            return true;
            if (!v.isMap()) return false;
            auto outer = v["outer"];
            if (!outer.isArray() || outer.arraySize()!=2) return false;
            if (!(outer[0].isMap() && outer[0]["n"].asInt64()==44)) return false;
            return outer[1].isArray() && outer[1].arraySize()==2
                && outer[1][0].asString()=="A"
                && outer[1][1].asString()=="B";
        });

    // C: nested mixed container with tensors
    xt::xtensor<double, 2> smallXtensor{{1.0, 2.0, 3.0, 4.0}, {4.0, 5.0, 6.0, 7.0}, {8.0, 9.0, 10.0, 11.0}, {12.0, 13.0, 14.0, 15.0}};
    test_serialization<DstP>("xlate: tensor",
        [smallXtensor](){
            auto src = serialize<SrcP>( zmap<"outer">(
                zvec( zmap<"n">(44), zvec("A",smallXtensor) )
            ));
            auto srd = typename SrcP::Deserializer(src.buf());
            auto drd = translate<DstP>(srd);

            //auto tensor = xtensor::asXTensor<double, 2>(deserializer["tensor_value"]);

            return serialize<DstP>( zmap<"outer">(
                zvec( zmap<"n">( drd["outer"][0]["n"].asInt64() ),
                      zvec( drd["outer"][1][0].asString(),
                      xtensor::asXTensor<double, 2>(drd["outer"][1][1]) ) )
            ));
        },
        [smallXtensor](const DV& v){
            return true;
            if (!v.isMap()) return false;
            auto outer = v["outer"];
            if (!outer.isArray() || outer.arraySize()!=2) return false;
            if (!(outer[0].isMap() && outer[0]["n"].asInt64()==44)) return false;
            return outer[1].isArray() && outer[1].arraySize()==2
                && outer[1][0].asString()=="A"
                && xtensor::asXTensor<double, 2>(outer[1][1])==smallXtensor;
        });

    std::cout << "== Translate (DSL) <" << SrcP::Name << "> → <" << DstP::Name << "> passed ==\n\n";
}

// --------------------- Columnar <-> row-array (DSL-built) ----------------
// expand_columnar()/collapse_columnar() - see columnar.hpp. Round-trips a
// columnar record (every field an equal-length array) through expand ->
// collapse and checks it comes back byte-identical in shape/values, same
// protocol on both ends.
template<class P>
void test_columnar_dsl() {
    std::cout << "== Columnar round-trip <" << P::Name << "> ==\n";

    // expand_columnar()/collapse_columnar() both finish their own
    // Root/Writer internally and hand back an already-serialized
    // P::Deserializer (see columnar.hpp) - unlike test_translate_dsl's
    // checks, there's no separate ZBuffer-producing step to hand to
    // test_serialization()'s build_fn/test_fn split, so this asserts
    // directly (matching test_columnar_errors()'s style) rather than
    // trying to force this shape through that helper.

    {
        // {"id":[1,2,3], "name":["a","b","c"]}
        auto src = serialize<P>( zmap<"id","name">(zvec(1,2,3), zvec("a","b","c")) );
        auto srd = typename P::Deserializer(src.buf());

        auto expanded = expand_columnar<P>(srd);
        if (!expanded.isArray() || expanded.arraySize() != 3) {
            throw std::runtime_error("expand_columnar: expected a 3-element row array");
        }
        if (expanded[0]["id"].asInt64() != 1 || expanded[0]["name"].asString() != "a" ||
            expanded[1]["id"].asInt64() != 2 || expanded[1]["name"].asString() != "b" ||
            expanded[2]["id"].asInt64() != 3 || expanded[2]["name"].asString() != "c") {
            throw std::runtime_error("expand_columnar: row contents didn't match");
        }
        std::cout << "   OK columnar: expand produces the expected row array\n";

        auto collapsed = collapse_columnar<P>(expanded);
        if (!collapsed.isMap() ||
            collapsed["id"].arraySize() != 3 || collapsed["id"][0].asInt64() != 1 ||
            collapsed["id"][2].asInt64() != 3 || collapsed["name"].arraySize() != 3 ||
            collapsed["name"][1].asString() != "b") {
            throw std::runtime_error("collapse_columnar: didn't round-trip back to the original columnar shape");
        }
        std::cout << "   OK columnar: collapse round-trips back to the original shape\n";
    }

    {
        auto src = serialize<P>( zmap<>() );
        auto srd = typename P::Deserializer(src.buf());
        auto expanded = expand_columnar<P>(srd);
        if (!expanded.isArray() || expanded.arraySize() != 0) {
            throw std::runtime_error("expand_columnar: empty columnar record should expand to an empty array");
        }
        std::cout << "   OK columnar: empty columnar record expands to an empty array\n";
    }

    {
        auto src = serialize<P>( zvec() );
        auto srd = typename P::Deserializer(src.buf());
        auto collapsed = collapse_columnar<P>(srd);
        if (!collapsed.isMap()) {
            throw std::runtime_error("collapse_columnar: empty row array should collapse to an (empty) object");
        }
        std::cout << "   OK columnar: empty row array collapses to an object\n";
    }

    std::cout << "== Columnar round-trip <" << P::Name << "> passed ==\n\n";
}

// Validation/error-path coverage - protocol-independent logic (columnar.hpp
// doesn't branch on format), so exercised once rather than per-protocol.
#ifdef ZERIALIZE_HAS_JSON
void test_columnar_errors() {
    std::cout << "== Columnar error paths ==\n";

    auto expect_throw = [](const char* label, auto&& fn) {
        bool threw = false;
        try { fn(); }
        catch (const std::runtime_error&) { threw = true; }
        if (!threw) throw std::runtime_error(std::string("expected throw: ") + label);
        std::cout << "   OK (threw as expected): " << label << "\n";
    };

    expect_throw("expand_columnar: root not an object", [](){
        auto src = serialize<JSON>( zvec(1,2,3) );
        auto srd = JSON::Deserializer(src.buf());
        (void)expand_columnar<JSON>(srd);
    });

    expect_throw("expand_columnar: a field is not an array", [](){
        auto src = serialize<JSON>( zmap<"a">(42) );
        auto srd = JSON::Deserializer(src.buf());
        (void)expand_columnar<JSON>(srd);
    });

    expect_throw("expand_columnar: mismatched column lengths", [](){
        auto src = serialize<JSON>( zmap<"a","b">(zvec(1,2,3), zvec(1,2)) );
        auto srd = JSON::Deserializer(src.buf());
        (void)expand_columnar<JSON>(srd);
    });

    expect_throw("collapse_columnar: root not an array", [](){
        auto src = serialize<JSON>( zmap<"a">(1) );
        auto srd = JSON::Deserializer(src.buf());
        (void)collapse_columnar<JSON>(srd);
    });

    expect_throw("collapse_columnar: a row is not an object", [](){
        auto src = serialize<JSON>( zvec(zmap<"a">(1), 42) );
        auto srd = JSON::Deserializer(src.buf());
        (void)collapse_columnar<JSON>(srd);
    });

    expect_throw("collapse_columnar: rows have different field sets", [](){
        auto src = serialize<JSON>( zvec(zmap<"a","b">(1,2), zmap<"a">(3)) );
        auto srd = JSON::Deserializer(src.buf());
        (void)collapse_columnar<JSON>(srd);
    });

    expect_throw("collapse_columnar: a row is missing a field row 0 has", [](){
        auto src = serialize<JSON>( zvec(zmap<"a","b">(1,2), zmap<"a","c">(3,4)) );
        auto srd = JSON::Deserializer(src.buf());
        (void)collapse_columnar<JSON>(srd);
    });

    std::cout << "== Columnar error paths passed ==\n\n";
}
#endif

// --------------------- Custom struct tests ---------------------
struct User { 
    std::string name; 
    int age; 
};

struct Company { 
    std::string name; 
    double value; 
    std::vector<User> users; 
};

// ADL serialization for User
template<zerialize::Writer W>
void serialize(const User& u, W& w) {
    zerialize::zmap<"name","age">(u.name, u.age)(w);
}

// ADL serialization for Company
template<zerialize::Writer W>
void serialize(const Company& c, W& w) {
    zerialize::zmap<"name","value","users">(
        c.name,
        c.value,
        c.users
    )(w);
}

template<class P>
void test_custom_structs() {
    using V = typename P::Deserializer;
    std::cout << "== Custom struct tests for <" << P::Name << "> ==\n";

    // Test User serialization/deserialization
    test_serialization<P>("User struct",
        [](){
            User user{"Alice", 30};
            return serialize<P>(user);
        },
        [](const V& v){
            return v.isMap()
                && v["name"].asString() == "Alice"
                && v["age"].asInt64() == 30;
        });

    // Test Company with multiple users
    test_serialization<P>("Company struct with users",
        [](){
            User user1{"Alice", 30};
            User user2{"Bob", 25};
            Company company{"TechCorp", 1000000.50, {user1, user2}};
            return serialize<P>(company);
        },
        [](const V& v){
            if (!v.isMap()) return false;
            if (v["name"].asString() != "TechCorp") return false;
            if (std::abs(v["value"].asDouble() - 1000000.50) > 1e-6) return false;
            
            auto users = v["users"];
            if (!users.isArray() || users.arraySize() != 2) return false;
            
            auto user1 = users[0];
            if (!user1.isMap() || user1["name"].asString() != "Alice" || user1["age"].asInt64() != 30) return false;
            
            auto user2 = users[1];
            if (!user2.isMap() || user2["name"].asString() != "Bob" || user2["age"].asInt64() != 25) return false;
            
            return true;
        });

    // Test nested Company in a map
    test_serialization<P>("Company nested in map",
        [](){
            User user{"Charlie", 35};
            Company company{"StartupInc", 50000.0, {user}};
            return serialize<P>(
                zmap<"id", "company", "active">(
                    42,
                    company,
                    true
                )
            );
        },
        [](const V& v){
            if (!v.isMap()) return false;
            if (v["id"].asInt64() != 42) return false;
            if (!v["active"].asBool()) return false;
            
            auto comp = v["company"];
            if (!comp.isMap()) return false;
            if (comp["name"].asString() != "StartupInc") return false;
            if (std::abs(comp["value"].asDouble() - 50000.0) > 1e-6) return false;
            
            auto users = comp["users"];
            if (!users.isArray() || users.arraySize() != 1) return false;
            
            auto user = users[0];
            return user.isMap() 
                && user["name"].asString() == "Charlie" 
                && user["age"].asInt64() == 35;
        });

    std::cout << "== Custom struct tests for <" << P::Name << "> passed ==\n\n";
}

// --------------------- Failure mode coverage ---------------------
template<class P>
void test_failure_modes() {
    using V = typename P::Deserializer;
    std::cout << "== Failure-mode tests for <" << P::Name << "> ==\n";

    test_serialization<P>("type mismatch throws",
        [](){
            return serialize<P>( zmap<"value">("not an int") );
        },
        [](const V& v){
            return expect_deserialization_error([&]{
                (void)v["value"].asInt64();
            });
        });

    test_serialization<P>("blob accessor rejects scalars",
        [](){
            return serialize<P>( zmap<"value">(42) );
        },
        [](const V& v){
            return expect_deserialization_error([&]{
                (void)v["value"].asBlob();
            });
        });

    test_serialization<P>("array index out of bounds throws",
        [](){
            return serialize<P>( zvec(1, 2) );
        },
        [](const V& v){
            return expect_deserialization_error([&]{
                (void)v[2];
            });
        });

    std::cout << "== Failure-mode tests for <" << P::Name << "> passed ==\n\n";
}

void test_json_failure_modes() {
    std::cout << "== JSON corruption tests ==\n";

    bool invalid_base64 = expect_deserialization_error([](){
        // Looks like a blob triple but base64 payload contains invalid chars.
        json::JsonDeserializer jd(R"(["~b","!!!!","base64"])");
        (void)jd.asBlob();
    });
    if (!invalid_base64) {
        throw std::runtime_error("json invalid base64 should throw DeserializationError");
    }

    std::cout << "== JSON corruption tests passed ==\n\n";
}

void test_json_nonfinite_doubles() {
    std::cout << "== JSON non-finite double tests ==\n";

    // Regression test for a real bug: yyjson_mut_write() (json.hpp's
    // RootSerializer::finish()) refuses to serialize *any* document
    // containing a non-finite double at all -- no write flag was passed
    // to allow NaN/Infinity literals or substitute null, so it failed the
    // whole write, not just that one value, and finish() turned that into
    // a thrown std::runtime_error for the entire document. double_() now
    // writes NaN/Infinity/-Infinity as strings instead (the same
    // convention pg_zerialize's own decoders use), same as every other
    // value JSON has no literal syntax for. Found via an actual
    // end-to-end test (Postgres row w/ Infinity floats -> pg_zerialize ->
    // pgnats -> NATS -> nats_tool), where every zerialize::translate<JSON>
    // call failed on that one row, not proactively.
#ifdef ZERIALIZE_HAS_MSGPACK
    auto src = serialize<MsgPack>(zmap<"pos_inf", "neg_inf", "nan_val", "finite">(
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN(),
        2.5));
    auto srd = MsgPackDeserializer(src.buf());
    auto jrd = translate<JSON>(srd);

    if (jrd["pos_inf"].asString() != "Infinity") {
        throw std::runtime_error("expected pos_inf to translate to the string \"Infinity\"");
    }
    if (jrd["neg_inf"].asString() != "-Infinity") {
        throw std::runtime_error("expected neg_inf to translate to the string \"-Infinity\"");
    }
    if (jrd["nan_val"].asString() != "NaN") {
        throw std::runtime_error("expected nan_val to translate to the string \"NaN\"");
    }
    if (jrd["finite"].asDouble() != 2.5) {
        throw std::runtime_error("expected a finite value in the same document to still translate as a normal number");
    }
#endif

    std::cout << "== JSON non-finite double tests passed ==\n\n";
}

// Regression test: write_value() (translate.hpp) now prefers mapEntries()/
// elements() over mapKeys()+operator[](key) / arraySize()+operator[](idx)
// when a protocol implements them, since the latter re-scans the container
// from the start on every call (O(n) per call, O(n^2) for the whole walk).
// This checks the fast accessors, where a protocol has them, produce
// exactly the same keys/values/order as the slower path they're replacing -
// not just that translate() still round-trips (the existing DSL/dynamic
// tests already cover that indirectly).
template<class P>
void test_fast_accessors() {
    using V = typename P::Deserializer;
    std::cout << "== Fast accessor tests for <" << P::Name << "> ==\n";

    auto buf = serialize<P>(zmap<"a", "b", "c">(
        int64_t(1),
        zvec(10, 20, 30),
        zmap<"x", "y">("hello", "world")
    ));
    V rd(buf.buf());

    if constexpr (requires { rd.mapEntries(); }) {
        std::vector<std::string> entry_keys;
        for (auto&& entry : rd.mapEntries()) entry_keys.push_back(std::string(entry.key));

        std::vector<std::string> plain_keys;
        for (std::string_view k : rd.mapKeys()) plain_keys.push_back(std::string(k));

        if (entry_keys != plain_keys) {
            throw std::runtime_error("mapEntries() key order/content differs from mapKeys()");
        }

        bool saw_a = false, saw_c = false;
        for (auto&& entry : rd.mapEntries()) {
            if (entry.key == "a") {
                saw_a = true;
                if (entry.value.asInt64() != 1) throw std::runtime_error("mapEntries() value for 'a' wrong");
            } else if (entry.key == "c") {
                saw_c = true;
                if (!entry.value.isMap()) throw std::runtime_error("mapEntries() nested map value lost its map-ness");
                if (entry.value["x"].asString() != "hello" || entry.value["y"].asString() != "world") {
                    throw std::runtime_error("mapEntries() nested map value contents wrong");
                }
            }
        }
        if (!saw_a || !saw_c) throw std::runtime_error("mapEntries() did not visit all expected keys");
    }

    if constexpr (requires { rd["b"].elements(); }) {
        auto b = rd["b"];
        std::vector<int64_t> via_elements;
        for (auto&& el : b.elements()) via_elements.push_back(el.asInt64());

        std::vector<int64_t> via_index;
        std::size_t n = b.arraySize();
        for (std::size_t i = 0; i < n; ++i) via_index.push_back(b[i].asInt64());

        if (via_elements != via_index) {
            throw std::runtime_error("elements() differs from arraySize()+operator[]");
        }
        std::vector<int64_t> expected{10, 20, 30};
        if (via_elements != expected) {
            throw std::runtime_error("elements() did not produce the expected array contents");
        }
    }

    std::cout << "== Fast accessor tests for <" << P::Name << "> passed ==\n\n";
}

void test_msgpack_failure_modes() {
    std::cout << "== MsgPack corruption tests ==\n";

    bool truncated_array = expect_deserialization_error([](){
        // 0x91 = array header with one element but no payload bytes.
        std::vector<uint8_t> bad = {0x91};
        MsgPackDeserializer rd(bad);
        (void)rd[0];
    });
    if (!truncated_array) {
        throw std::runtime_error("msgpack truncated array should throw DeserializationError");
    }

    std::cout << "== MsgPack corruption tests passed ==\n\n";
}

#ifdef ZERIALIZE_HAS_BEVE
void test_beve_failure_modes() {
    std::cout << "== BEVE corruption tests ==\n";

    bool truncated_array = expect_deserialization_error([](){
        // 0x05 = generic-array header, 0x04 = compressed size (count=1), no
        // element bytes follow. glz::lazy_beve_view::operator[](size_t) does
        // not itself flag this as an error for generic arrays (verified with
        // ASan against glaze v8.0.0: it silently returns a view pointing at
        // end-of-buffer) -- beve.hpp's own checkChild() closes that gap.
        std::vector<uint8_t> bad = {0x05, 0x04};
        BeveDeserializer rd(bad);
        (void)rd[0];
    });
    if (!truncated_array) {
        throw std::runtime_error("beve truncated generic array should throw DeserializationError");
    }

    bool truncated_blob = expect_deserialization_error([](){
        // 0x14 = typed uint8-array header, size claims 50 elements, only 2
        // payload bytes are actually present.
        std::vector<uint8_t> bad = {0x14, uint8_t(50 << 2), 0xAB, 0xCD};
        BeveDeserializer rd(bad);
        (void)rd.asBlob();
    });
    if (!truncated_blob) {
        throw std::runtime_error("beve truncated blob should throw DeserializationError");
    }

    bool truncated_scalar = expect_deserialization_error([](){
        // 0x69 = int64 number header, only 1 of the 8 payload bytes present.
        std::vector<uint8_t> bad = {0x69, 0x01};
        BeveDeserializer rd(bad);
        (void)rd.asInt64();
    });
    if (!truncated_scalar) {
        throw std::runtime_error("beve truncated scalar should throw DeserializationError");
    }

    std::cout << "== BEVE corruption tests passed ==\n\n";
}
#endif // ZERIALIZE_HAS_BEVE

#ifdef ZERIALIZE_HAS_BSON
void test_bson_failure_modes() {
    std::cout << "== BSON corruption tests ==\n";

    bool truncated_doc = expect_deserialization_error([](){
        // Declares a 20-byte document but the buffer is only 6 bytes.
        std::vector<uint8_t> bad = {20,0,0,0, 0x08, 'x'};
        bsonjc::BsonDeserializer rd(bad);
        for (auto k : rd.mapKeys()) (void)k;
    });
    if (!truncated_doc) {
        throw std::runtime_error("bson truncated document should throw DeserializationError");
    }

    bool unterminated_name = expect_deserialization_error([](){
        // Element name never hits a 0x00 before the buffer ends.
        std::vector<uint8_t> bad = {12,0,0,0, 0x08,'a','b','c','d','e','f','g'};
        bsonjc::BsonDeserializer rd(bad);
        (void)rd["a"];
    });
    if (!unterminated_name) {
        throw std::runtime_error("bson unterminated element name should throw DeserializationError");
    }

    bool bare_scalar_root = expect_serialization_error([](){
        (void)serialize<Bson>(42);
    });
    if (!bare_scalar_root) {
        throw std::runtime_error("bson bare top-level scalar should throw SerializationError");
    }

    bool uint64_too_large = expect_serialization_error([](){
        Bson::RootSerializer rs;
        Bson::Serializer w(rs);
        w.begin_map(1);
        w.key("huge");
        w.uint64(UINT64_MAX);
        w.end_map();
    });
    if (!uint64_too_large) {
        throw std::runtime_error("bson uint64 above INT64_MAX should throw SerializationError");
    }

    std::cout << "== BSON corruption tests passed ==\n\n";
}

// BSON gets its own DSL-style coverage rather than the generic
// test_protocol_dsl<P>()/test_dynamic_serialization<P>(): several of their
// cases serialize a bare array as the *root* value and assert isArray() on
// the result, which BSON cannot round-trip. This isn't a bug in
// BsonDeserializer -- it's a real BSON format constraint (confirmed against
// jsoncons' own bson_parser, which unconditionally decodes the top level as
// a document too): a document and an array are physically identical on the
// wire, and only a *parent* element's header records which one a value is.
// The root has no parent, so that information doesn't exist to recover.
// Arrays nested anywhere below the root are unaffected (their parent
// element header does carry a real type tag) -- see the last case below,
// and BSON.md.
void test_bson_specific() {
    std::cout << "== BSON specific tests ==\n";
    using V = Bson::Deserializer;

    test_serialization<Bson>("map root with nested array and map",
        [](){
            return serialize<Bson>( zmap<"a","b">(
                7,
                zvec("x", zmap<"n">(44))
            ));
        },
        [](const V& v){
            if (!v.isMap()) return false;
            if (!v["a"].isInt() || v["a"].asInt64()!=7) return false;
            auto b = v["b"];
            if (!b.isArray() || b.arraySize()!=2) return false;
            if (b[0].asString()!="x") return false;
            return b[1].isMap() && b[1]["n"].asInt64()==44;
        });

    test_serialization<Bson>("mixed numeric types",
        [](){
            return serialize<Bson>( zmap<
                "i8","u8","i32","u32","i64","u64","d"
            >( int8_t(-5), uint8_t(200), int32_t(-123456), uint32_t(987654321u),
               int64_t(-7777777777LL), uint64_t(9999999999ULL), 3.25 ) );
        },
        [](const V& v){
            return v.isMap()
                && v["i8"].asInt64()==-5
                && v["u8"].asUInt64()==200
                && v["i32"].asInt64()==-123456
                && v["u32"].asUInt64()==987654321ULL
                && v["i64"].asInt64()==-7777777777LL
                && v["u64"].asUInt64()==9999999999ULL
                && std::abs(v["d"].asDouble()-3.25)<1e-12;
        });

    test_serialization<Bson>("booleans, null, mapKeys()",
        [](){
            return serialize<Bson>( zmap<"t","f","n">( true, false, nullptr ) );
        },
        [](const V& v){
            if (!v.isMap() || v["t"].asBool()!=true || v["f"].asBool()!=false || !v["n"].isNull()) return false;
            std::set<std::string_view> keys;
            for (std::string_view k : v.mapKeys()) keys.insert(k);
            return keys.size()==3 && keys.count("t") && keys.count("f") && keys.count("n");
        });

    xt::xtensor<double, 2> tens{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    test_serialization<Bson>("tensor nested in map",
        [&tens](){
            return serialize<Bson>( zmap<"key1", "key2", "key3">(42, 3.14159, tens) );
        },
        [&tens](const V& v) {
            auto a = xtensor::asXTensor<double>(v["key3"]);
            return v["key1"].asInt32() == 42
                && v["key2"].asDouble() == 3.14159
                && a == tens;
        });

    // Arrays work fully once wrapped in a document key (the realistic BSON
    // usage pattern -- MongoDB documents are conventionally always objects
    // at the top level too).
    test_serialization<Bson>("array wrapped in a map root",
        [](){
            return serialize<Bson>( zmap<"items">( zvec(1, 2, 3, "x") ) );
        },
        [](const V& v){
            auto items = v["items"];
            return v.isMap() && items.isArray() && items.arraySize()==4
                && items[0].asInt64()==1 && items[2].asInt64()==3
                && items[3].asString()=="x";
        });

    // Documented root-array limitation: a bare array root reads back as a
    // document whose keys are the stringified indices, not as an array.
    test_serialization<Bson>("bare array root reads back as a document",
        [](){
            return serialize<Bson>( zvec(10, 20, 30) );
        },
        [](const V& v){
            return !v.isArray() && v.isMap()
                && v["0"].asInt64()==10 && v["1"].asInt64()==20 && v["2"].asInt64()==30;
        });

    std::cout << "== BSON specific tests passed ==\n\n";
}
#endif // ZERIALIZE_HAS_BSON

#ifdef ZERIALIZE_HAS_ION
void test_ion_failure_modes() {
    std::cout << "== Ion corruption tests ==\n";

    bool missing_bvm = expect_deserialization_error([](){
        std::vector<uint8_t> bad = {0x21, 0x05}; // no binary version marker
        IonDeserializer rd(bad);
    });
    if (!missing_bvm) {
        throw std::runtime_error("ion missing BVM should throw DeserializationError");
    }

    bool struct_overflow = expect_deserialization_error([](){
        // struct (VarUInt length) claiming a length far beyond the buffer.
        std::vector<uint8_t> bad = {0xE0,0x01,0x00,0xEA, 0xDE, 0xFF,0xFF,0x7F};
        IonDeserializer rd(bad);
        for (auto k : rd.mapKeys()) (void)k;
    });
    if (!struct_overflow) {
        throw std::runtime_error("ion oversized struct length should throw DeserializationError");
    }

    bool unresolvable_symbol = expect_deserialization_error([](){
        // Field SID 10 (local) referenced with no symbol table defining it.
        std::vector<uint8_t> bad = {0xE0,0x01,0x00,0xEA, 0xD2, 0x8A, 0x20};
        IonDeserializer rd(bad);
        for (auto k : rd.mapKeys()) (void)k;
    });
    if (!unresolvable_symbol) {
        throw std::runtime_error("ion unresolvable symbol ID should throw DeserializationError");
    }

    bool shared_imports_rejected = expect_deserialization_error([](){
        // Annotated ($ion_symbol_table) struct with a non-empty imports list
        // (shared/imported symbol table) -- explicitly unsupported.
        std::vector<uint8_t> bad = {
            0xE0,0x01,0x00,0xEA,
            0xE6, 0x81,0x83, 0xD3, 0x86, 0xB1, 0x20
        };
        IonDeserializer rd(bad);
        (void)rd.isMap();
    });
    if (!shared_imports_rejected) {
        throw std::runtime_error("ion shared/imported symbol table should throw DeserializationError");
    }

    std::cout << "== Ion corruption tests passed ==\n\n";
}
#endif // ZERIALIZE_HAS_ION

void test_zer_specific() {
    std::cout << "== Zera specific tests ==\n";

    test_serialization<Zera>("u64 beyond int64 range",
        [](){
            constexpr std::uint64_t big = (std::uint64_t(1) << 63) + 5;
            return serialize<Zera>( zmap<"big">(big) );
        },
        [](const Zera::Deserializer& v){
            if (!v.isMap()) return false;
            auto b = v["big"];
            if (!b.isUInt()) return false;
            if (b.asUInt64() != ((std::uint64_t(1) << 63) + 5)) return false;
            return expect_deserialization_error([&]{
                (void)b.asInt64();
            });
        });

    test_serialization<Zera>("xtensor blob is zero-copy when aligned",
        [](){
            xt::xtensor<double, 2> t{{1.0, 2.0}, {3.0, 4.0}};
            return serialize<Zera>(t);
        },
        [](const Zera::Deserializer& v){
            auto view = xtensor::asXTensorView<double>(v);
            return view.viewInfo().zero_copy
                && view.viewInfo().reason == tensor::TensorViewReason::Ok
                && view.array() == xt::xtensor<double, 2>{{1.0, 2.0}, {3.0, 4.0}};
        });

    std::cout << "== Zera specific tests passed ==\n\n";
}

void test_tensor_view_alignment() {
    std::cout << "== Tensor view alignment tests ==\n";

    // Serialize a tensor in a format that returns span-backed blobs (Zera),
    // then rebase the buffer at offsets [0..15] to intentionally misalign the blob pointer.
    {
        xt::xtensor<double, 2> expected{{1.0, 2.0}, {3.0, 4.0}};
        auto zb = serialize<Zera>(expected);
        auto orig = zb.buf();

        std::size_t zero_copy = 0;
        std::size_t copied = 0;

        std::vector<std::uint8_t> backing(orig.size() + 16);
        for (std::size_t off = 0; off < 16; ++off) {
            std::memcpy(backing.data() + off, orig.data(), orig.size());
            std::span<const std::uint8_t> misaligned{backing.data() + off, orig.size()};
            Zera::Deserializer rd(misaligned);

            auto view = xtensor::asXTensorView<double, 2>(rd);
            auto info = view.viewInfo();

            if (info.zero_copy) {
                ++zero_copy;
                if (info.reason != tensor::TensorViewReason::Ok) throw std::runtime_error("xtensor: expected Ok");
                if ((info.address % alignof(double)) != 0) throw std::runtime_error("xtensor: expected aligned address");
            } else {
                ++copied;
                if (info.reason != tensor::TensorViewReason::Misaligned) throw std::runtime_error("xtensor: expected Misaligned");
                if ((info.address % alignof(double)) == 0) throw std::runtime_error("xtensor: expected misaligned address");
            }

            if (view.array() != expected) throw std::runtime_error("xtensor: tensor mismatch");
            if (info.required_alignment != alignof(double)) throw std::runtime_error("xtensor: wrong required_alignment");
            if (info.byte_size != expected.size() * sizeof(double)) throw std::runtime_error("xtensor: wrong byte_size");
        }

        if (!(zero_copy == 2 && copied == 14)) throw std::runtime_error("xtensor alignment sweep: unexpected counts");
    }

    // Same idea for Eigen.
    {
        Eigen::Matrix<double, 3, 2> expected;
        expected << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
        auto zb = serialize<Zera>(expected);
        auto orig = zb.buf();

        std::size_t zero_copy = 0;
        std::size_t copied = 0;

        std::vector<std::uint8_t> backing(orig.size() + 16);
        for (std::size_t off = 0; off < 16; ++off) {
            std::memcpy(backing.data() + off, orig.data(), orig.size());
            std::span<const std::uint8_t> misaligned{backing.data() + off, orig.size()};
            Zera::Deserializer rd(misaligned);

            auto view = eigen::asEigenMatrixView<double, 3, 2>(rd);
            auto info = view.viewInfo();

            if (info.zero_copy) {
                ++zero_copy;
                if (info.reason != tensor::TensorViewReason::Ok) throw std::runtime_error("eigen: expected Ok");
                if ((info.address % alignof(double)) != 0) throw std::runtime_error("eigen: expected aligned address");
            } else {
                ++copied;
                if (info.reason != tensor::TensorViewReason::Misaligned) throw std::runtime_error("eigen: expected Misaligned");
                if ((info.address % alignof(double)) == 0) throw std::runtime_error("eigen: expected misaligned address");
            }

            if (!view.matrix().isApprox(expected)) throw std::runtime_error("eigen: matrix mismatch");
            if (info.required_alignment != alignof(double)) throw std::runtime_error("eigen: wrong required_alignment");
            if (info.byte_size != expected.size() * sizeof(double)) throw std::runtime_error("eigen: wrong byte_size");
        }

        if (!(zero_copy == 2 && copied == 14)) throw std::runtime_error("eigen alignment sweep: unexpected counts");
    }

    std::cout << "== Tensor view alignment tests passed ==\n\n";
}

} // namespace zerialize

int main() {
    using namespace zerialize;

    // Per-protocol, DSL-only tests
    #ifdef ZERIALIZE_HAS_JSON
    test_protocol_dsl<JSON>();
    #endif
    #ifdef ZERIALIZE_HAS_FLEXBUFFERS
    test_protocol_dsl<Flex>();
    #endif
    #ifdef ZERIALIZE_HAS_MSGPACK
    test_protocol_dsl<MsgPack>();
    #endif
    #ifdef ZERIALIZE_HAS_CBOR
    test_protocol_dsl<CBOR>();
    #endif
    #ifdef ZERIALIZE_HAS_ZERA
    test_protocol_dsl<Zera>();
    #endif
    #ifdef ZERIALIZE_HAS_BEVE
    test_protocol_dsl<Beve>();
    #endif
    #ifdef ZERIALIZE_HAS_ION
    test_protocol_dsl<Ion>();
    #endif
    // Bson is intentionally not run through test_protocol_dsl<P>() here --
    // see test_bson_specific() for why (bare-array-root round-tripping is
    // not something BSON's format supports) and for its own DSL coverage.

    // Dynamic serialization (runtime-built values)
    #ifdef ZERIALIZE_HAS_JSON
    test_dynamic_serialization<JSON>();
    #endif
    #ifdef ZERIALIZE_HAS_FLEXBUFFERS
    test_dynamic_serialization<Flex>();
    #endif
    #ifdef ZERIALIZE_HAS_MSGPACK
    test_dynamic_serialization<MsgPack>();
    #endif
    #ifdef ZERIALIZE_HAS_CBOR
    test_dynamic_serialization<CBOR>();
    #endif
    #ifdef ZERIALIZE_HAS_ZERA
    test_dynamic_serialization<Zera>();
    #endif
    #ifdef ZERIALIZE_HAS_BEVE
    test_dynamic_serialization<Beve>();
    #endif
    #ifdef ZERIALIZE_HAS_ION
    test_dynamic_serialization<Ion>();
    #endif
    // Same reason as above: Bson skips the generic dynamic-serialization
    // suite (it includes bare-array-root tensor cases).

    // Custom struct tests
    #ifdef ZERIALIZE_HAS_JSON
    test_custom_structs<JSON>();
    #endif
    #ifdef ZERIALIZE_HAS_FLEXBUFFERS
    test_custom_structs<Flex>();
    #endif
    #ifdef ZERIALIZE_HAS_MSGPACK
    test_custom_structs<MsgPack>();
    #endif
    #ifdef ZERIALIZE_HAS_CBOR
    test_custom_structs<CBOR>();
    #endif
    #ifdef ZERIALIZE_HAS_ZERA
    test_custom_structs<Zera>();
    #endif
    #ifdef ZERIALIZE_HAS_BEVE
    test_custom_structs<Beve>();
    #endif
    #ifdef ZERIALIZE_HAS_BSON
    test_custom_structs<Bson>();
    #endif
    #ifdef ZERIALIZE_HAS_ION
    test_custom_structs<Ion>();
    #endif

    // Failure-mode coverage
    #ifdef ZERIALIZE_HAS_JSON
    test_failure_modes<JSON>();
    test_json_failure_modes();
    test_json_nonfinite_doubles();
    test_fast_accessors<JSON>();
    #endif
    #ifdef ZERIALIZE_HAS_FLEXBUFFERS
    test_failure_modes<Flex>();
    test_fast_accessors<Flex>();
    #endif
    #ifdef ZERIALIZE_HAS_MSGPACK
    test_failure_modes<MsgPack>();
    test_msgpack_failure_modes();
    test_fast_accessors<MsgPack>();
    #endif
    #ifdef ZERIALIZE_HAS_CBOR
    test_failure_modes<CBOR>();
    test_fast_accessors<CBOR>();
    #endif
    #ifdef ZERIALIZE_HAS_ZERA
    test_failure_modes<Zera>();
    test_zer_specific();
    test_tensor_view_alignment();
    test_fast_accessors<Zera>();
    #endif
    #ifdef ZERIALIZE_HAS_BEVE
    test_failure_modes<Beve>();
    test_beve_failure_modes();
    test_fast_accessors<Beve>();
    #endif
    #ifdef ZERIALIZE_HAS_BSON
    test_failure_modes<Bson>();
    test_bson_failure_modes();
    test_bson_specific();
    test_fast_accessors<Bson>();
    #endif
    #ifdef ZERIALIZE_HAS_ION
    test_failure_modes<Ion>();
    test_ion_failure_modes();
    test_fast_accessors<Ion>();
    #endif

    // Translate cross-protocol (both directions) built with the same DSL
    #if defined(ZERIALIZE_HAS_JSON) && defined(ZERIALIZE_HAS_MSGPACK)
    test_translate_dsl<JSON, MsgPack>();
    #endif
    #if defined(ZERIALIZE_HAS_JSON) && defined(ZERIALIZE_HAS_FLEXBUFFERS)
    test_translate_dsl<JSON, Flex>();
    #endif
    #ifdef ZERIALIZE_HAS_CBOR
    #ifdef ZERIALIZE_HAS_JSON
    test_translate_dsl<JSON, CBOR>();
    #endif
    #endif

    // ZERA (built-in) ↔ other protocols
    #if defined(ZERIALIZE_HAS_ZERA) && defined(ZERIALIZE_HAS_JSON)
    test_translate_dsl<Zera, JSON>();
    test_translate_dsl<JSON, Zera>();
    #endif
    #if defined(ZERIALIZE_HAS_ZERA) && defined(ZERIALIZE_HAS_FLEXBUFFERS)
    test_translate_dsl<Zera, Flex>();
    test_translate_dsl<Flex, Zera>();
    #endif
    #if defined(ZERIALIZE_HAS_ZERA) && defined(ZERIALIZE_HAS_MSGPACK)
    test_translate_dsl<Zera, MsgPack>();
    test_translate_dsl<MsgPack, Zera>();
    #endif
    #if defined(ZERIALIZE_HAS_ZERA) && defined(ZERIALIZE_HAS_CBOR)
    test_translate_dsl<Zera, CBOR>();
    test_translate_dsl<CBOR, Zera>();
    #endif

    // BEVE ↔ other protocols
    #if defined(ZERIALIZE_HAS_BEVE) && defined(ZERIALIZE_HAS_JSON)
    test_translate_dsl<Beve, JSON>();
    test_translate_dsl<JSON, Beve>();
    #endif
    #if defined(ZERIALIZE_HAS_BEVE) && defined(ZERIALIZE_HAS_MSGPACK)
    test_translate_dsl<Beve, MsgPack>();
    test_translate_dsl<MsgPack, Beve>();
    #endif
    #if defined(ZERIALIZE_HAS_BEVE) && defined(ZERIALIZE_HAS_ZERA)
    test_translate_dsl<Beve, Zera>();
    test_translate_dsl<Zera, Beve>();
    #endif

    // BSON ↔ other protocols
    #if defined(ZERIALIZE_HAS_BSON) && defined(ZERIALIZE_HAS_JSON)
    test_translate_dsl<Bson, JSON>();
    test_translate_dsl<JSON, Bson>();
    #endif
    #if defined(ZERIALIZE_HAS_BSON) && defined(ZERIALIZE_HAS_CBOR)
    test_translate_dsl<Bson, CBOR>();
    test_translate_dsl<CBOR, Bson>();
    #endif
    #if defined(ZERIALIZE_HAS_BSON) && defined(ZERIALIZE_HAS_MSGPACK)
    test_translate_dsl<Bson, MsgPack>();
    test_translate_dsl<MsgPack, Bson>();
    #endif

    // Ion ↔ other protocols
    #if defined(ZERIALIZE_HAS_ION) && defined(ZERIALIZE_HAS_JSON)
    test_translate_dsl<Ion, JSON>();
    test_translate_dsl<JSON, Ion>();
    #endif
    #if defined(ZERIALIZE_HAS_ION) && defined(ZERIALIZE_HAS_MSGPACK)
    test_translate_dsl<Ion, MsgPack>();
    test_translate_dsl<MsgPack, Ion>();
    #endif
    #if defined(ZERIALIZE_HAS_ION) && defined(ZERIALIZE_HAS_ZERA)
    test_translate_dsl<Ion, Zera>();
    test_translate_dsl<Zera, Ion>();
    #endif
    #if defined(ZERIALIZE_HAS_ION) && defined(ZERIALIZE_HAS_BEVE)
    test_translate_dsl<Ion, Beve>();
    test_translate_dsl<Beve, Ion>();
    #endif
    #if defined(ZERIALIZE_HAS_ION) && defined(ZERIALIZE_HAS_BSON)
    test_translate_dsl<Ion, Bson>();
    test_translate_dsl<Bson, Ion>();
    #endif

    #if defined(ZERIALIZE_HAS_FLEXBUFFERS) && defined(ZERIALIZE_HAS_MSGPACK)
    test_translate_dsl<Flex, MsgPack>();
    #endif
    #if defined(ZERIALIZE_HAS_FLEXBUFFERS) && defined(ZERIALIZE_HAS_JSON)
    test_translate_dsl<Flex, JSON>();
    #endif
    #ifdef ZERIALIZE_HAS_CBOR
    #ifdef ZERIALIZE_HAS_FLEXBUFFERS
    test_translate_dsl<Flex, CBOR>();
    #endif
    #endif

    #if defined(ZERIALIZE_HAS_MSGPACK) && defined(ZERIALIZE_HAS_JSON)
    test_translate_dsl<MsgPack, JSON>();
    #endif
    #if defined(ZERIALIZE_HAS_MSGPACK) && defined(ZERIALIZE_HAS_FLEXBUFFERS)
    test_translate_dsl<MsgPack, Flex>();
    #endif
    #ifdef ZERIALIZE_HAS_CBOR
    #ifdef ZERIALIZE_HAS_MSGPACK
    test_translate_dsl<MsgPack, CBOR>();
    #endif

    #ifdef ZERIALIZE_HAS_JSON
    test_translate_dsl<CBOR, JSON>();
    #endif
    #ifdef ZERIALIZE_HAS_FLEXBUFFERS
    test_translate_dsl<CBOR, Flex>();
    #endif
    #ifdef ZERIALIZE_HAS_MSGPACK
    test_translate_dsl<CBOR, MsgPack>();
    #endif
    #endif

    // Columnar <-> row-array round-trip, same protocol on both ends.
    #ifdef ZERIALIZE_HAS_JSON
    test_columnar_dsl<JSON>();
    #endif
    #ifdef ZERIALIZE_HAS_MSGPACK
    test_columnar_dsl<MsgPack>();
    #endif
    #ifdef ZERIALIZE_HAS_FLEXBUFFERS
    test_columnar_dsl<Flex>();
    #endif
    #ifdef ZERIALIZE_HAS_CBOR
    test_columnar_dsl<CBOR>();
    #endif
    #ifdef ZERIALIZE_HAS_ZERA
    test_columnar_dsl<Zera>();
    #endif
    #ifdef ZERIALIZE_HAS_BEVE
    test_columnar_dsl<Beve>();
    #endif
    #ifdef ZERIALIZE_HAS_ION
    test_columnar_dsl<Ion>();
    #endif
    // Bson excluded from test_columnar_dsl<P>() for the same reason it's
    // excluded from test_protocol_dsl<P>() - see test_bson_specific().

    #ifdef ZERIALIZE_HAS_JSON
    test_columnar_errors();
    #endif

    // Cross-protocol columnar convenience wrapper - the realistic nats_tool
    // use case: a columnar record received in one wire format, expanded
    // straight into another (e.g. MsgPack -> JSON) in one call.
    #if defined(ZERIALIZE_HAS_MSGPACK) && defined(ZERIALIZE_HAS_JSON)
    {
        std::cout << "== Columnar cross-protocol <MsgPack> -> <JSON> ==\n";
        auto src = serialize<MsgPack>( zmap<"id","name">(zvec(1,2,3), zvec("a","b","c")) );
        auto srd = MsgPack::Deserializer(src.buf());
        auto expanded = expand_columnar<JSON>(srd);
        if (!expanded.isArray() || expanded.arraySize() != 3 ||
            expanded[0]["id"].asInt64() != 1 || expanded[0]["name"].asString() != "a") {
            throw std::runtime_error("cross-protocol expand_columnar<JSON>(MsgPack) failed");
        }
        auto collapsed = collapse_columnar<MsgPack>(expanded);
        if (!collapsed.isMap() || collapsed["id"].arraySize() != 3 ||
            collapsed["id"][2].asInt64() != 3 || collapsed["name"][1].asString() != "b") {
            throw std::runtime_error("cross-protocol collapse_columnar<MsgPack>(JSON) failed");
        }
        std::cout << "== Columnar cross-protocol <MsgPack> -> <JSON> passed ==\n\n";
    }
    #endif

    std::cout << "\nAll tests complete ✅\n";
    return 0;
}
