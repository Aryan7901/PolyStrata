// src/poly/inkling2_dspark.cpp - Inkling 2's DSpark drafter (--draft-model): Thinking Machines' drafter for the model,
// the official mtp.safetensors as a "dflash" GGUF.  Adapted from Basalt's src/models/inkling2/dspark.{hpp,cpp}
// (https://github.com/jesdga95/basalt, MIT, Copyright the Basalt authors), whose forward follows vLLM's Inkling 2
// DSpark draft (vllm/models/inkling2/nvidia/dspark.py).
//
// What the target leaves (ds_capture(), outputs()): for every row it runs, the inputs of three of its runs (5, 23 and
// 41: layer 5, the loop's layer 11 in its second pass, its layer 17 in its third) as bf16 taps, and its final normed
// state.  Every row of every call goes in, right or not: a block writes its own cells over the positions it covers,
// so a wrong row's are never read.
//
//   context   mix_l = rms(sum_k softmax(fuse[l])_k f_k) * dflare_norm[l] for draft layer l, and the Markov head's
//             f = rms(sum_k softmax(fuse[3])_k f_k) * dspark_norm.  Position p's keys and values in layer l:
//             k = rope(rms_head(k_ctx mix_l) * k_ctx_norm, p), v = v_ctx mix_l, into the layer's ring at p.
//   block     anchor a (the target's pick, not fed yet) at position p, rows p .. p + 7: x0 = rms(embed(a)) *
//             token_embd_norm (the target's normed embedding), x1..7 = mask_embd.  Each layer: y = attn_norm(x); q,
//             k, v of y, q and k normed per head and turned at their rows' positions (NeoX); the block's keys and
//             values go to the ring at p .. p + 7; every row attends the layer's window before it and all eight block
//             rows, scale 1/sqrt(dim); x += o; x += down(silu(gate z) * up z), z = ffn_norm(x).  h = norm(x).
//   head      draft i (from block row i + 1) in turn: c_i = rnn_b + W_h h_{i+1} + W_f f (f of the row at p - 1);
//             raw = c_i + W_sp [state ; w1[prev]] (state 0 and prev = a before draft 0); gate = sigmoid(raw[0:R]),
//             state = gate * state + (1 - gate) * tanh(raw[R:2R]), latent = tanh(raw[2R:3R]);
//             logits = softcap(head(h_{i+1}) + fold latent + fold_b), fold = head x hidden_out and fold_b = head x
//             hidden_out's bias, made once; the draft is the first argmax, its chance the softmax probability.
//
// Numerics (a draft only changes the speed): the block's products are the file's Q8_0 rounded to Q4_0 as they load,
// the context's stay Q8_0, and a block's rows go through MMQ together; the head is the target's Q4_0 copy of its
// output matrix (the shortlist's) and the fold a Q4_0 of it; the recurrence's hidden and context terms are Q8_0, its
// state's and token's BF16 with fp32 activations.  Unlike Basalt's, the full layer reads a window too
// (INK_DSPARK_SPAN positions, 8192 by default): its whole context would take VRAM from the experts of an 8 GB card.
#include "inkling2.hpp"

#include "mmq.hpp"
#include "runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include "ggml.h"

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>   // ranges for nsys: the block and the commit
#define DS_RANGE(name) nvtxRangePushA(name)
#define DS_RANGE_END() nvtxRangePop()
#else
#define DS_RANGE(name) ((void) 0)
#define DS_RANGE_END() ((void) 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::poly {
namespace {

constexpr int kFoldChunk = 4096;   // the output's rows a step of the fold takes

float bf16_value(uint16_t u) {
    const uint32_t w = (uint32_t) u << 16;
    float f;
    std::memcpy(&f, &w, sizeof f);
    return f;
}

}  // namespace

void Inkling2::ds_geometry() {
    const Shards& g = *dg_;
    const std::string& file = opt_.draft_model;
    const auto need = [&](bool ok, const std::string& what) {
        if (!ok) throw std::runtime_error(file + " is not a DSpark drafter of this model: " + what);
    };
    const MetaValue* arch = g.get("general.architecture");
    need(arch && arch->s == "dflash", "its architecture is not dflash");
    const auto val = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = meta_in(g, key);
        need(v != nullptr, std::string("it has no dflash.") + key);
        return *v;
    };
    const auto num = [&](const char* key) {
        const MetaValue& v = val(key);
        need(v.is_num(), std::string("dflash.") + key + " is not a number");
        return v.num();
    };
    Dspark& d = ds_;
    d.layers = (int) num("block_count");
    d.n_ff = (int) num("feed_forward_length");
    d.heads = (int) num("attention.head_count");
    d.kv_heads = (int) num("attention.head_count_kv");
    d.dim = (int) num("attention.key_length");
    d.block = (int) num("block_size");
    d.rank = (int) num("dspark.markov_rank");
    d.window = (int) num("attention.sliding_window");
    d.eps = (float) num("attention.layer_norm_rms_epsilon");
    const double theta = num("rope.freq_base");
    const MetaValue* rot = meta_in(g, "rope.dimension_count");
    const MetaValue* cap = meta_in(g, "final_logit_softcapping");
    d.cap = cap ? (float) cap->num() : 0.f;
    need((int) num("embedding_length") == n_embd_, "its width is not the model's");
    need((int) num("vocab_size") == n_vocab_, "its vocabulary is not the model's");
    need(d.layers == 3, "it has other than three layers");
    need(d.block >= 2 && d.block <= 8, "a block of other than 2 to 8 rows");
    need(d.heads > 0 && d.kv_heads > 0 && d.heads % d.kv_heads == 0 && d.dim % 32 == 0 && d.dim <= 256,
         "heads this engine does not run");
    need((int) num("attention.value_length") == d.dim, "keys and values of different lengths");
    need(!rot || (int) rot->num() == d.dim, "rotary positions on part of a head");
    need(d.rank % 32 == 0 && d.rank <= 1024, "a Markov rank that is not a multiple of 32");
    need(d.n_ff % 32 == 0 && d.window > 0, "a bad feed_forward_length or window");
    const MetaValue* anchor = meta_in(g, "sample_from_anchor");
    need(!anchor || anchor->u == 0, "it drafts from the anchor's own row");
    const MetaValue& tl = val("target_layers");
    need(tl.type == MetaType::ARRAY && tl.items.size() == 4 && (int) tl.items[3].u == n_layer_,
         "its targets are not three of the model's runs and its output");
    for (int k = 0; k < 3; ++k) {
        d.targets[k] = (int) tl.items[(size_t) k].u;
        need(d.targets[k] >= 0 && d.targets[k] < n_layer_ && (k == 0 || d.targets[k] > d.targets[k - 1]),
             "its target runs are out of order");
    }
    const MetaValue& pat = val("attention.sliding_window_pattern");
    need(pat.type == MetaType::ARRAY && (int) pat.items.size() == d.layers, "no window pattern for its layers");
    d.theta_scale = (float) std::pow(theta, -2.0 / d.dim);
    int span = 8192;
    if (const char* s = std::getenv("INK_DSPARK_SPAN")) span = std::atoi(s);
    d.span = std::clamp(span, d.window, std::max(d.window, opt_.max_context));
    d.L.assign((size_t) d.layers, DsLayer{});
    for (int l = 0; l < d.layers; ++l) {
        DsLayer& D = d.L[(size_t) l];
        D.swa = pat.items[(size_t) l].u != 0;
        D.window = D.swa ? d.window : d.span;   // the positions before the block its rows read
        D.cells = D.window + 64;                 // and its rows, with room
    }
    d.vocab = draft_rows();
    if (const char* pr = std::getenv("INK_DSPARK_PROF")) d.prof = std::atoi(pr) == 1;
    if (const char* id = std::getenv("INK_DSPARK_IDLE")) d.idle = std::atoi(id) == 1;
    if (const char* dr = std::getenv("INK_DSPARK_DRY")) d.dry = std::atoi(dr) == 1;
    if (opt_.batch > 0 || !opt_.layer_split.empty()) {
        std::fprintf(stderr, "strata-poly: --draft-model: the DSpark drafter guesses for one sequence on one card; "
                             "off with --batch and --layer-split\n");
        dg_.reset();
        block_ = nullptr;
        return;
    }
    d.on = true;
    std::fprintf(stderr, "strata-poly: DSpark drafter: %d layers reading runs %d, %d and %d and the output, blocks of %d, "
                         "the full layer over the last %d positions, drafts among the first %d tokens\n",
                 d.layers, d.targets[0], d.targets[1], d.targets[2], d.block, d.span, d.vocab);
}

void Inkling2::ds_load() {
    Dspark& d = ds_;
    const Shards& g = *dg_;
    const int E = n_embd_, Q = d.heads * d.dim, KV = d.kv_heads * d.dim, R = d.rank;
    const size_t arena_at = dense_at_;
    struct Count {   // the arena's bytes this call added, once the call is done
        Dspark& d;
        const size_t& at;
        size_t from;
        ~Count() { d.vram_weights = at - from; }
    } count{d, dense_at_, arena_at};
    const auto t = [&](const std::string& n) {
        const TensorInfo* x = g.find(n);
        if (!x) throw std::runtime_error(opt_.draft_model + " has no tensor " + n);
        return x;
    };
    const auto f32 = [&](const std::string& n, int64_t a) { return upload(t(n), n, a, 1, kF32); };
    const auto mat = [&](const std::string& n, int64_t a, int64_t b, Want w) { return upload(t(n), n, a, b, w); };
    for (int l = 0; l < d.layers; ++l) {
        DsLayer& D = d.L[(size_t) l];
        const std::string b = "blk." + std::to_string(l) + ".";
        D.wq = mat(b + "attn_q.weight", E, Q, kGuessMat);
        D.wk = mat(b + "attn_k.weight", E, KV, kGuessMat);
        D.wv = mat(b + "attn_v.weight", E, KV, kGuessMat);
        D.wo = mat(b + "attn_output.weight", Q, E, kGuessMat);
        D.gate = mat(b + "ffn_gate.weight", E, d.n_ff, kGuessMat);
        D.up = mat(b + "ffn_up.weight", E, d.n_ff, kGuessMat);
        D.down = mat(b + "ffn_down.weight", d.n_ff, E, kGuessMat);
        D.k_ctx = mat(b + "attn_k_ctx.weight", E, KV, kAnyMat);
        D.v_ctx = mat(b + "attn_v_ctx.weight", E, KV, kAnyMat);
        D.attn_norm = f32(b + "attn_norm.weight", E);
        D.post_norm = f32(b + "ffn_norm.weight", E);
        D.dflare_norm = f32(b + "dflare_norm.weight", E);
        D.q_norm = f32(b + "attn_q_norm.weight", d.dim);
        D.k_norm = f32(b + "attn_k_norm.weight", d.dim);
        D.k_ctx_norm = f32(b + "attn_k_ctx_norm.weight", d.dim);
    }
    d.mask = f32("mask_embd.weight", E);
    d.norm = f32("output_norm.weight", E);
    d.markov_norm = f32("dspark_norm.weight", E);
    d.rnn_b = f32("joint_proj.bias", 3 * R);
    d.out_b = f32("dspark_hidden_out.bias", E);
    const auto shaped = [&](const TensorInfo* x, uint32_t type, int64_t n_in, int64_t n_out) {
        if (x->type != type || (int64_t) x->shape[0] != n_in || (int64_t) x->elements() != n_in * n_out)
            throw std::runtime_error(opt_.draft_model + ": " + x->name + " is " + shape_of(*x) + ", not [" +
                                     std::to_string(n_in) + ", " + std::to_string(n_out) + "]");
        return x;
    };
    const TensorInfo* jp = shaped(t("joint_proj.weight"), kTypeBF16, 2 * R + 2 * E, 3 * R);
    shaped(t("dspark_hidden_out.weight"), kTypeBF16, R, E);
    const TensorInfo* w1 = shaped(t("markov_w1.weight"), kTypeBF16, R, n_vocab_);
    const TensorInfo* fw = shaped(t("dspark_fuse.weight"), kTypeF32, 4, 4);
    const TensorInfo* mk = t("mask_embd.weight");
    // the arena's own: the recurrence's projection split in three - its state's and token's columns (BF16, the
    // Markov step reads them), its hidden's and its context's (Q8_0, products of a block's rows) - the fuse weights
    // softmaxed, the head's fold (Q4_0) and the fold of the bias
    const auto arena = [&](size_t bytes) {
        uint8_t* p = dense_ ? dense_ + dense_at_ : nullptr;
        dense_at_ += pad(bytes) + kTail;
        if (dense_ && dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for the drafter");
        return p;
    };
    const size_t in = 2 * (size_t) R + 2 * (size_t) E, out = 3 * (size_t) R, q8_row = ggml_row_size(GGML_TYPE_Q8_0, E);
    d.w_sp = Dev{arena(out * 2 * R * 2), kTypeBF16, 2 * R, (int) out};
    d.w_h = Dev{arena(out * q8_row), (uint32_t) GGML_TYPE_Q8_0, E, (int) out};
    d.w_f = Dev{arena(out * q8_row), (uint32_t) GGML_TYPE_Q8_0, E, (int) out};
    d.alpha = Dev{arena(16 * sizeof(float)), kTypeF32, 4, 4};
    d.fold = Dev{arena((size_t) d.vocab * ggml_row_size(GGML_TYPE_Q4_0, R)), (uint32_t) GGML_TYPE_Q4_0, R, d.vocab};
    d.fold_b = Dev{arena((size_t) d.vocab * sizeof(float)), kTypeF32, d.vocab, 1};
    if (!dense_) return;
    {
        const uint16_t* src = (const uint16_t*) g.data(*jp);
        std::vector<uint16_t> a(out * 2 * R);
        std::vector<float> h(out * E), f(out * E);
        for (size_t o = 0; o < out; ++o) {
            std::memcpy(a.data() + o * 2 * R, src + o * in, 2 * (size_t) R * sizeof(uint16_t));
            for (int e = 0; e < E; ++e) {
                h[o * E + (size_t) e] = bf16_value(src[o * in + 2 * R + (size_t) e]);
                f[o * E + (size_t) e] = bf16_value(src[o * in + 2 * R + (size_t) E + (size_t) e]);
            }
        }
        std::vector<uint8_t> hq(out * q8_row), fq(out * q8_row);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, h.data(), hq.data(), 0, (int64_t) out, E, nullptr);
        ggml_quantize_chunk(GGML_TYPE_Q8_0, f.data(), fq.data(), 0, (int64_t) out, E, nullptr);
        h2d(d.w_sp.p, a.data(), a.size() * sizeof(uint16_t));
        h2d(d.w_h.p, hq.data(), hq.size());
        h2d(d.w_f.p, fq.data(), fq.size());
    }
    {   // a softmax over the four features of each mix, in double
        float raw[16], al[16];
        std::memcpy(raw, g.data(*fw), sizeof raw);
        for (int j = 0; j < 4; ++j) {
            double m = raw[j * 4], z = 0;
            for (int k = 1; k < 4; ++k) m = std::max(m, (double) raw[j * 4 + k]);
            for (int k = 0; k < 4; ++k) z += std::exp((double) raw[j * 4 + k] - m);
            for (int k = 0; k < 4; ++k) al[j * 4 + k] = (float) (std::exp((double) raw[j * 4 + k] - m) / z);
        }
        h2d(d.alpha.p, al, sizeof al);
    }
    if (mk->type != kTypeF32) throw std::runtime_error(opt_.draft_model + ": mask_embd.weight is not F32");
    d.h_mask.assign((const float*) g.data(*mk), (const float*) g.data(*mk) + E);
    if (!d.w1) {   // the Markov head's token rows in pinned host memory: a step reads one row over the bus
        const size_t bytes = (size_t) n_vocab_ * R * sizeof(uint16_t);
        ck(cudaHostAlloc(&d.w1_host, bytes, cudaHostAllocMapped), "the drafter's token rows in host memory");
        std::memcpy(d.w1_host, g.data(*w1), bytes);
        void* dev = nullptr;
        ck(cudaHostGetDevicePointer(&dev, d.w1_host, 0), "the drafter's token rows' card address");
        d.w1 = (uint16_t*) dev;
    }
    ck(cudaStreamSynchronize(cs(st_)), "the drafter's weights");   // the host vectors above go now
}

void Inkling2::ds_scratch() {
    Dspark& d = ds_;
    const size_t scratch_from = scratch_at_;
    const size_t E = (size_t) n_embd_, Q = (size_t) d.heads * d.dim, KV = (size_t) d.kv_heads * d.dim,
                 B = (size_t) d.block, R = (size_t) d.rank, V = (size_t) d.vocab;
    d.tap = (uint16_t*) fbuf((3 * (size_t) rows_ * E + 1) / 2);
    d.mix = fbuf(3 * kDsCommit * E);
    d.fr = fbuf(kDsRing * E);
    d.kb = fbuf(kDsCommit * KV);
    d.vb = fbuf(kDsCommit * KV);
    d.kr = fbuf(kDsCommit * KV);
    d.x = fbuf(B * E);
    d.xn = fbuf(B * E);
    d.q = fbuf(B * Q);
    d.qr = fbuf(B * Q);
    d.att = fbuf(B * Q);
    d.tmp = fbuf(B * E);
    d.g = fbuf(B * (size_t) d.n_ff);
    d.u = fbuf(B * (size_t) d.n_ff);
    d.hh = fbuf(B * E);
    int most = 0;
    for (const DsLayer& D : d.L) most = std::max(most, D.window);
    // ds_attend()'s partial sums; at load the fold's columns
    d.score_floats = std::max(ds_attend_floats(d.block, d.heads, d.kv_heads, d.dim, (int64_t) most + d.block), R * E);
    d.score = fbuf(d.score_floats);
    d.base = fbuf(std::max((B - 1) * V, R * (size_t) kFoldChunk));   // at load the fold's rows of a step
    d.ctx = fbuf((B - 1) * 3 * R);
    d.ctx_f = fbuf(3 * R);
    d.state = fbuf(2 * R);
    d.latent = fbuf(R);
    d.vram_scratch = (scratch_at_ - scratch_from) * sizeof(float);
    if (d.lk) return;
    d.lk = (Likeliest*) dmalloc(B * sizeof(Likeliest), "the drafts");
    d.prev = (int32_t*) dmalloc(sizeof(int32_t), "the drafter's anchor");
    d.h_lk = (Likeliest*) hmalloc(B * sizeof(Likeliest), "the drafts");
    d.h_in = (float*) hmalloc((B * E + 1) * sizeof(float), "the drafter's rows");
    d.pick = dmalloc(ds_pick_bytes(), "the drafter's pick");
    for (DsLayer& D : d.L) {
        const size_t bytes = (size_t) D.cells * KV * sizeof(uint16_t);
        D.kc = (uint16_t*) dmalloc(bytes, "the drafter's keys");
        D.vc = (uint16_t*) dmalloc(bytes, "the drafter's values");
        zero_vram(D.kc, bytes);
        zero_vram(D.vc, bytes);
        d.vram_own += 2 * bytes;
    }
    d.vram_own += ds_pick_bytes();
}

// The fold of the target's head (the Q4_0 copy the drafts read) with the Markov head's output projection, and of its
// bias: fold[v] = head[v] . hidden_out, fold_b[v] = head[v] . hidden_out's bias.  Once, before the first block.
void Inkling2::ds_fold() {
    Dspark& d = ds_;
    const int E = n_embd_, R = d.rank, V = d.vocab;
    Dev head = head_short_ ? out_q4_ : output_;
    const size_t row = ggml_row_size((ggml_type) head.type, E), frow = ggml_row_size(GGML_TYPE_Q4_0, R);
    const TensorInfo* ho = dg_->find("dspark_hidden_out.weight");
    const uint16_t* w = (const uint16_t*) dg_->data(*ho);   // E rows of R
    std::vector<float> wt((size_t) R * E);                   // its columns as rows
    for (int e = 0; e < E; ++e)
        for (int r = 0; r < R; ++r) wt[(size_t) r * E + e] = bf16_value(w[(size_t) e * R + r]);
    h2d(d.score, wt.data(), wt.size() * sizeof(float));
    for (int v0 = 0; v0 < V; v0 += kFoldChunk) {
        const int c = std::min(kFoldChunk, V - v0);
        Dev part = head;
        part.p = (uint8_t*) head.p + (size_t) v0 * row;
        part.n_out = c;
        lin(part, d.score, d.base, R);   // R rows of c: column r of the chunk's rows of the fold
        ds_fold_q4(d.base, c, R, (uint8_t*) d.fold.p + (size_t) v0 * frow, st_);
    }
    head.n_out = V;
    lin(head, fp(d.out_b), (float*) d.fold_b.p, 1);
    ck(cudaStreamSynchronize(cs(st_)), "the drafter's fold");
    d.folded = true;
}

// The run's input rows into its tap, when it is one of the drafter's targets.  A read that takes its calls a layer at
// a time runs this run for every call before the output of any: the call's rows wait in host memory.
void Inkling2::ds_capture(int il, const float* cur, int rs, int n) {
    int slot = -1;
    for (int k = 0; k < 3; ++k)
        if (ds_.targets[k] == il) slot = k;
    if (slot < 0 || rows_of_) return;
    const size_t E = (size_t) n_embd_;
    uint16_t* dst = ds_.tap + ((size_t) slot * rows_ + (size_t) ds_.tap_off) * E;
    ds_tap(cur, rs, dst, n_embd_, n, st_);
    if (read_at_ < 0) return;
    if (ds_.h_tap_rows < read_rows_) {
        hfree(ds_.h_tap);
        ds_.h_tap = nullptr;
        ds_.h_tap = (uint16_t*) hmalloc(3 * (size_t) read_rows_ * E * sizeof(uint16_t), "the drafter's taps");
        ds_.h_tap_rows = read_rows_;
    }
    ck(cudaMemcpyAsync(ds_.h_tap + ((size_t) slot * read_rows_ + (size_t) (pos_ - read_at_)) * E, dst,
                       (size_t) n * E * sizeof(uint16_t), cudaMemcpyDeviceToHost, cs(st_)),
       "the drafter's taps");
}

void Inkling2::shift_rows(ptrdiff_t by) {
    Engine::shift_rows(by);
    ds_.tap_off += by;
}

// keys and values of n positions from pos into a layer's ring
void Inkling2::ds_store(const DsLayer& D, const float* k, const float* v, int64_t pos, int n) {
    const int KV = ds_.kv_heads * ds_.dim;
    const int a = (int) (pos % D.cells), head = std::min(n, D.cells - a);
    store_f16(k, D.kc, KV, a, head, st_);
    store_f16(v, D.vc, KV, a, head, st_);
    if (head < n) {
        store_f16(k + (size_t) head * KV, D.kc, KV, 0, n - head, st_);
        store_f16(v + (size_t) head * KV, D.vc, KV, 0, n - head, st_);
    }
}

// A product of a few rows where its exactness does not matter: the tensor cores' MMQ reads the matrix once for all of
// them (MMVQ's exact layout for several columns read it at a third of the bus's speed for the head's seven rows).
void Inkling2::ds_mm(const Dev& w, const float* x, float* y, int n) {
    if (n < 2 || !mmq_ || !Mmq::fits((int) w.type, w.n_out, w.n_in)) return lin(w, x, y, n);
    Mmq::quantize(x, nullptr, mqx_, (int) w.type, w.n_in, w.n_in, n, st_);
    mmq_->mul(w.p, (int) w.type, w.n_out, w.n_in, 0, 1, mqx_, nullptr, nullptr, n, n, y, w.n_out, st_);
    if (w.scale != 1.0f) axpy(y, w.scale, y, n * w.n_out, false, st_);
}

// The context of the rows a call ran: the mixes of their features, each layer's keys and values of them.
void Inkling2::outputs(int64_t pos, int n, const float* h) {
    if (!ds_.on || n <= 0) return;
    Dspark& d = ds_;
    const size_t E = (size_t) n_embd_;
    if (read_at_ >= 0)   // the call's taps back from host memory
        for (int k = 0; k < 3; ++k)
            ck(cudaMemcpyAsync(d.tap + (size_t) k * rows_ * E,
                               d.h_tap + ((size_t) k * read_rows_ + (size_t) (pos - read_at_)) * E,
                               (size_t) n * E * sizeof(uint16_t), cudaMemcpyHostToDevice, cs(st_)),
               "the drafter's taps");
    DS_RANGE("dspark commit");
    const bool timed = d.prof && n <= d.block;   // INK_DSPARK_PROF: an answer's calls, not a prompt's
    if (timed) ck(cudaStreamSynchronize(cs(st_)), "the drafter's commit");
    const Clock::time_point c0 = Clock::now();
    DsMix a;
    a.alpha = fp(d.alpha);
    for (int l = 0; l < d.layers; ++l) a.norm[l] = fp(d.L[(size_t) l].dflare_norm);
    a.norm[3] = fp(d.markov_norm);
    a.eps = d.eps;
    for (int r0 = 0; r0 < n; r0 += kDsCommit) {
        const int m = std::min(kDsCommit, n - r0);
        ds_mix(d.tap + (size_t) r0 * E, (int64_t) rows_ * (int64_t) E, h + (size_t) r0 * E, a, n_embd_, pos + r0, d.mix,
               m, d.fr, kDsRing, st_);
        for (int l = 0; l < d.layers; ++l) {
            const DsLayer& D = d.L[(size_t) l];
            const float* x = d.mix + (size_t) l * m * E;
            ds_mm(D.k_ctx, x, d.kb, m);
            ds_mm(D.v_ctx, x, d.vb, m);
            head_norm_rope(d.kb, d.dim, fp(D.k_ctx_norm), d.kr, d.dim, d.kv_heads, d.dim, d.theta_scale, pos + r0, m,
                           d.eps, st_);
            ds_store(D, d.kr, d.vb, pos + r0, m);
        }
    }
    if (timed) {
        ck(cudaStreamSynchronize(cs(st_)), "the drafter's commit");
        d.ph[7] += ms_since(c0);
        ++d.commits;
    }
    // the record of the positions the caches hold: a gap or a jump back starts it again
    if (pos > d.hi || pos + n < d.lo) d.lo = pos;
    else d.lo = std::min(d.lo, pos);
    d.hi = pos + n;
    DS_RANGE_END();
}

int Inkling2::draft_block(int32_t token, int n, int32_t* ids, float* chance, const std::function<bool(int)>* more) {
    Dspark& d = ds_;
    if (!d.on || d.idle || n < 1) return 0;
    if (token < 0 || token >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    const int64_t p = pos_;
    // the Markov head's context is the row before the anchor's: kept for the last kDsRing positions run
    if (p < 1 || p - 1 >= d.hi || p - 1 < std::max(d.lo, d.hi - kDsRing) || p + d.block > opt_.max_context) return 0;
    if (!d.folded) ds_fold();
    DS_RANGE("dspark block");
    const Clock::time_point t0 = Clock::now();
    Clock::time_point at = t0;
    const auto mark = [&](int phase) {   // INK_DSPARK_PROF
        if (!d.prof) return;
        ck(cudaStreamSynchronize(cs(st_)), "the drafter");
        d.ph[phase] += ms_since(at);
        at = Clock::now();
    };
    if (d.prof) ck(cudaStreamSynchronize(cs(st_)), "the drafter");
    n = std::min(n, d.block - 1);
    const size_t E = (size_t) n_embd_, B = (size_t) d.block, R = (size_t) d.rank, V = (size_t) d.vocab;
    // the block's rows: the anchor's embedding, normed as the model norms its tokens, then the mask
    embed(token, d.h_in);
    for (size_t r = 1; r < B; ++r) std::memcpy(d.h_in + r * E, d.h_mask.data(), E * sizeof(float));
    std::memcpy(d.h_in + B * E, &token, sizeof token);
    ck(cudaMemcpyAsync(d.xn, d.h_in, B * E * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "the drafter's rows");
    ck(cudaMemcpyAsync(d.prev, d.h_in + B * E, sizeof(int32_t), cudaMemcpyHostToDevice, cs(st_)), "the anchor");
    rms_norm(d.xn, fp(tok_norm_), d.x, n_embd_, 1, rms_eps_, st_);
    ink_rows(d.xn + E, n_embd_, d.x + E, n_embd_, n_embd_, d.block - 1, false, st_);
    const float scale = 1.f / std::sqrt((float) d.dim);
    mark(0);
    for (int l = 0; l < d.layers; ++l) {
        const DsLayer& D = d.L[(size_t) l];
        rms_norm(d.x, fp(D.attn_norm), d.xn, n_embd_, d.block, d.eps, st_);
        ds_mm(D.wq, d.xn, d.q, d.block);
        ds_mm(D.wk, d.xn, d.kb, d.block);
        ds_mm(D.wv, d.xn, d.vb, d.block);
        head_norm_rope(d.q, d.dim, fp(D.q_norm), d.qr, d.dim, d.heads, d.dim, d.theta_scale, p, d.block, d.eps, st_);
        head_norm_rope(d.kb, d.dim, fp(D.k_norm), d.kr, d.dim, d.kv_heads, d.dim, d.theta_scale, p, d.block, d.eps,
                       st_);
        ds_store(D, d.kr, d.vb, p, d.block);
        mark(1);
        ds_attend(d.qr, D.kc, D.vc, d.block, d.heads, d.kv_heads, d.dim, std::max(d.lo, p - D.window),
                  p + d.block, D.cells, scale, d.score, d.att, st_);
        mark(2);
        ds_mm(D.wo, d.att, d.tmp, d.block);
        axpy(d.tmp, 1.f, d.x, d.block * n_embd_, true, st_);
        rms_norm(d.x, fp(D.post_norm), d.xn, n_embd_, d.block, d.eps, st_);
        ds_mm(D.gate, d.xn, d.g, d.block);
        ds_mm(D.up, d.xn, d.u, d.block);
        swiglu_clamp(d.g, d.u, d.g, d.block * d.n_ff, INFINITY, st_);
        ds_mm(D.down, d.g, d.tmp, d.block);
        axpy(d.tmp, 1.f, d.x, d.block * n_embd_, true, st_);
        mark(3);
    }
    rms_norm(d.x, fp(d.norm), d.hh, n_embd_, d.block, d.eps, st_);
    // the head: the output matrix's term of every draft in one read of it, the recurrence's hidden and context terms
    Dev head = head_short_ ? out_q4_ : output_;
    head.n_out = d.vocab;
    ds_mm(head, d.hh + E, d.base, n);
    mark(4);
    // c_i = rnn_b + W_h h_{i+1} + W_f f
    ds_mm(d.w_h, d.hh + E, d.ctx, n);
    lin(d.w_f, d.fr + (size_t) ((p - 1) % kDsRing) * E, d.ctx_f, 1);
    add_bias(d.ctx_f, d.ctx, 3 * d.rank, n, st_);
    add_bias(fp(d.rnn_b), d.ctx, 3 * d.rank, n, st_);
    ck(cudaMemsetAsync(d.state, 0, R * sizeof(float), cs(st_)), "the drafter's state");
    const int cand = std::min(d.vocab, n_unpadded_);
    mark(5);
    // The drafts in turn.  With `more`, each one's pick is read back as soon as it is made, and the drafts stop when
    // no window could hold the next: the recurrence makes them one by one anyway.  The first two go together.
    int made = 0;
    for (int i = 0; i < n; ++i) {
        const int32_t* prev = i == 0 ? d.prev : &d.lk[i - 1].token;
        ds_markov(d.ctx + (size_t) i * 3 * R, (const uint16_t*) d.w_sp.p, d.w1, prev, d.rank, d.state + (i & 1) * R,
                  d.state + ((i + 1) & 1) * R, d.latent, st_);
        ds_pick(d.base + (size_t) i * V, d.fold.p, d.latent, d.rank, fp(d.fold_b), d.cap, cand, d.pick, d.lk + i, st_);
        const bool ask = more && i >= 1 && i + 1 < n;
        if (!ask && i + 1 < n) continue;
        ck(cudaMemcpyAsync(d.h_lk + made, d.lk + made, (size_t) (i + 1 - made) * sizeof(Likeliest), cudaMemcpyDeviceToHost,
                           cs(st_)),
           "the drafts");
        ck(cudaStreamSynchronize(cs(st_)), "the drafts");
        for (; made <= i; ++made) {
            ids[made] = d.h_lk[made].token;
            chance[made] = d.h_lk[made].prob;
        }
        if (ask && !(*more)(i)) break;
    }
    mark(6);
    n = made;
    ++d.blocks;
    d.drafted += n;
    d.block_ms += ms_since(t0);
    DS_RANGE_END();
    return d.dry ? 0 : n;
}

std::string Inkling2::draft_report() const {
    if (!ds_.on || ds_.blocks == 0) return {};
    char s[400];
    int at = std::snprintf(s, sizeof s, "%lld blocks of %.1f drafts, %.2f ms a block", ds_.blocks,
                           (double) ds_.drafted / (double) ds_.blocks, ds_.block_ms / (double) ds_.blocks);
    if (ds_.prof) {
        const double b = (double) ds_.blocks;
        std::snprintf(s + at, sizeof s - (size_t) at,
                      " (inputs %.3f, layers %.3f + attention %.3f + %.3f, head %.3f, context terms %.3f, drafts %.3f ms; "
                      "commit %.3f ms a call)",
                      ds_.ph[0] / b, ds_.ph[1] / b, ds_.ph[2] / b, ds_.ph[3] / b, ds_.ph[4] / b, ds_.ph[5] / b,
                      ds_.ph[6] / b, ds_.commits ? ds_.ph[7] / (double) ds_.commits : 0.0);
    }
    return s;
}

}  // namespace strata::poly
