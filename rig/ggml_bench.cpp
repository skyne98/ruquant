// Today's path, measured: ggml's mul_mat with q4_0 weights x f32 activations on the first GPU
// backend ggml finds (ROCm on the MI50, CUDA on the 3090), at the cost-model shapes and verify
// widths, plus the lm_head at q6_K. This is what gfx906/model.py's "today" column predicts.
//
// Weights rotate through enough copies to overflow L2. "us" is per mul_mat in a graph of
// several back to back (one per weight copy, at least 8 when they fit), the way a forward pass
// runs them: it includes ggml quantizing the activations to q8_1 for each mul_mat, and
// amortizes the graph launch and sync. "us_alone" is a graph of one mul_mat, synchronized.
//   ./ggml_bench [gpu index] [--quick] [--types w4a4,q4_0] [--lm q6_K,q8_0] [--widths 1,5,8] [--tp N]   -> JSON on stdout
// --types replaces the 27b linears' type (w4a4 turns into w4a4h for the shapes that read an online Hadamard
// input: attn.o / dn.out, mlp.down), --lm the lm_head's; the fn and drafter shapes are skipped with --types.
// --tp N times one GPU's share under N-way tensor parallelism: K / N for the row-parallel shapes (.o, down), N / N
// for the rest.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct Shape {
    const char* name;
    int K, N;
    ggml_type type;
};

static const Shape SHAPES[] = {
    {"27b attn.q+gate", 5120, 12288, GGML_TYPE_Q4_0}, {"27b attn.k/v", 5120, 1024, GGML_TYPE_Q4_0},
    {"27b attn.o / dn.out", 6144, 5120, GGML_TYPE_Q4_0}, {"27b dn.qkvz", 5120, 16384, GGML_TYPE_Q4_0},
    {"27b mlp.gate/up", 5120, 17408, GGML_TYPE_Q4_0}, {"27b mlp.down", 17408, 5120, GGML_TYPE_Q4_0},
    {"27b mlp.gate+up", 5120, 34816, GGML_TYPE_Q4_0}, // the two in one launch (ggml-cuda's W4A4 fusion)
    {"27b drafter q", 5120, 4096, GGML_TYPE_Q4_0}, {"27b drafter o", 4096, 5120, GGML_TYPE_Q4_0},
    {"27b drafter fc", 25600, 5120, GGML_TYPE_Q4_0}, {"fn qsa.q+gate", 2560, 12288, GGML_TYPE_Q4_0},
    {"fn dn.qkvz", 2560, 16384, GGML_TYPE_Q4_0}, {"fn qsa.o / dn.out", 6144, 2560, GGML_TYPE_Q4_0},
    {"fn expert gate/up", 2560, 640, GGML_TYPE_Q4_0}, {"fn expert down", 640, 2560, GGML_TYPE_Q4_0},
    {"27b lm_head", 5120, 248320, GGML_TYPE_Q6_K}, {"fn lm_head", 2560, 248320, GGML_TYPE_Q6_K},
};
static const int WIDTHS[] = {1, 4, 8, 16, 32, 64};

static std::vector<std::string> split(const char* s) {
    std::vector<std::string> out;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || *p == 0) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
            if (*p == 0) break;
        } else {
            cur += *p;
        }
    }
    return out;
}

static ggml_type type_by_name(const std::string& n) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char* tn = ggml_type_name((ggml_type)t);
        if (tn && n == tn) return (ggml_type)t;
    }
    std::fprintf(stderr, "unknown type %s\n", n.c_str());
    std::exit(1);
}

// Timing doesn't depend on the values: quantize 64 random rows once and tile them.
static std::vector<uint8_t> weights(const Shape& s, std::mt19937& rng) {
    const int rows = std::min(64, s.N);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> src((size_t)s.K * rows);
    for (auto& v : src) v = u(rng);
    const size_t row_bytes = ggml_row_size(s.type, s.K);
    std::vector<uint8_t> q(row_bytes * s.N);
    ggml_quantize_chunk(s.type, src.data(), q.data(), 0, rows, s.K, nullptr);
    for (int r = rows; r < s.N; ++r) std::memcpy(&q[r * row_bytes], &q[(r % rows) * row_bytes], row_bytes);
    return q;
}

struct Timing {
    double us, us_alone;
};

static double median_us(ggml_backend_t backend, ggml_cgraph* g, int reps) {
    std::vector<double> us;
    for (int i = 0; i < reps; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, g);
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}

static Timing point(ggml_backend_t backend, const Shape& s, int T, const std::vector<uint8_t>& q, std::mt19937& rng) {
    const size_t w_bytes = ggml_row_size(s.type, s.K) * s.N;
    const int copies = (int)std::max<size_t>(w_bytes < (256u << 20) ? 8 : 1, (64u << 20) / w_bytes + 1);

    ggml_init_params params{};
    params.mem_size = ggml_tensor_overhead() * (2 * copies + 2) + ggml_graph_overhead() * 2;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.K, T);
    std::vector<ggml_tensor*> w(copies);
    ggml_cgraph* all = ggml_new_graph(ctx);
    ggml_cgraph* one = ggml_new_graph(ctx);
    for (int c = 0; c < copies; ++c) {
        w[c] = ggml_new_tensor_2d(ctx, s.type, s.K, s.N);
        ggml_tensor* y = ggml_mul_mat(ctx, w[c], x);
        ggml_build_forward_expand(all, y);
        if (c == 0) ggml_build_forward_expand(one, y);
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return {-1, -1};
    }

    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (int c = 0; c < copies; ++c) ggml_backend_tensor_set(w[c], q.data(), 0, w_bytes);
    std::vector<float> xs((size_t)s.K * T);
    for (auto& v : xs) v = u(rng);
    ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * sizeof(float));

    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, all);
    const double us = median_us(backend, all, 10) / copies;
    for (int i = 0; i < 3; ++i) ggml_backend_graph_compute(backend, one);
    const double us_alone = median_us(backend, one, 20);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return {us, us_alone};
}

int main(int argc, char** argv) {
    const int want = argc > 1 && argv[1][0] != '-' ? std::atoi(argv[1]) : 0;
    bool quick = false;
    std::vector<std::string> types, lm;
    std::vector<int> widths(std::begin(WIDTHS), std::end(WIDTHS));
    int tp = 1;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 < argc && std::strcmp(argv[i], "--tp") == 0) tp = std::atoi(argv[i + 1]);
        quick |= std::strcmp(argv[i], "--quick") == 0;
        if (i + 1 < argc && std::strcmp(argv[i], "--types") == 0) types = split(argv[i + 1]);
        if (i + 1 < argc && std::strcmp(argv[i], "--lm") == 0) lm = split(argv[i + 1]);
        if (i + 1 < argc && std::strcmp(argv[i], "--widths") == 0) {
            widths.clear();
            for (auto& w : split(argv[i + 1])) widths.push_back(std::atoi(w.c_str()));
        }
    }

    ggml_backend_load_all();
    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0, gpu = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU && (int)gpu++ == want) {
            dev = d;
            break;
        }
    }
    if (!dev) {
        std::fprintf(stderr, "no GPU backend #%d (is ggml built with HIP / CUDA?)\n", want);
        return 1;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    std::printf("{\n  \"bench\": \"ggml\", \"device\": \"%s\", \"description\": \"%s\",\n  \"points\": [\n",
                ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));
    std::mt19937 rng(906);
    bool first = true;
    const int nshapes = (int)(sizeof(SHAPES) / sizeof(SHAPES[0]));
    std::vector<Shape> run;
    for (int i = 0; i < nshapes; ++i) {
        const Shape& s = SHAPES[i];
        if (quick && i != 0 && i != 14) continue;
        const bool head = s.type == GGML_TYPE_Q6_K;
        if (types.empty() && lm.empty()) { run.push_back(s); continue; }
        if (std::strncmp(s.name, "27b", 3) != 0 || std::strstr(s.name, "drafter")) continue;
        for (auto& t : head ? lm : types) {
            Shape v = s;
            v.type = type_by_name(t);
            if (v.type == GGML_TYPE_W4A4 && (std::strstr(s.name, ".o") || std::strstr(s.name, "down"))) v.type = GGML_TYPE_W4A4H;
            run.push_back(v);
        }
    }
    for (Shape& s : run) {
        if (std::strstr(s.name, ".o") || std::strstr(s.name, "down")) s.K /= tp; else s.N /= tp;
        const std::vector<uint8_t> q = weights(s, rng);
        for (int T : widths) {
            if (quick && T != 1 && T != 8 && T != 32) continue;
            const Timing t = point(backend, s, T, q, rng);
            const double w_bytes = (double)ggml_row_size(s.type, s.K) * s.N;
            std::printf("%s    {\"shape\": \"%s\", \"K\": %d, \"N\": %d, \"T\": %d, \"type\": \"%s\", \"us\": %.2f, "
                        "\"us_alone\": %.2f, \"weight_gbs\": %.1f}",
                        first ? "" : ",\n", s.name, s.K, s.N, T, ggml_type_name(s.type), t.us, t.us_alone,
                        t.us > 0 ? w_bytes / (t.us * 1e-6) / 1e9 : 0.0);
            first = false;
        }
    }
    std::printf("\n  ]\n}\n");
    ggml_backend_free(backend);
    return 0;
}
