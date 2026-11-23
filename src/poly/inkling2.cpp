// src/poly/inkling2.cpp - Inkling 2's layers.  The reference for the math is llama.cpp's inkling2 graph
// (src/models/inkling2.cpp, ggml-org/llama.cpp#32695 for the loop).
#include "inkling2.hpp"
#include "igpu.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>

namespace strata::poly {
namespace {

constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr float kMixEps = 1e-6f;                   // the graph's norms over the depth and in the loop's controllers
constexpr float kHeadMargin = 12.f;                // the output's shortlist: within this of the best (Q4_0) logit
constexpr int kHeadCap = 1024;                     // tokens a row's shortlist takes at most
constexpr int kV4Block = 64;                       // k8v4 and k4v4: the values are rotated in blocks of this many

int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

}  // namespace

// llama_hparams::loop_layer: a run's layer of the file
int Inkling2::file_layer(int il) const {
    if (passes_ < 2 || il < core_start_) return il;
    const int core = core_end_ - core_start_;
    if (il < core_start_ + passes_ * core) return core_start_ + (il - core_start_) % core;
    return il - (passes_ - 1) * core;
}

Dev Inkling2::fused(int il, std::initializer_list<const char*> names, int64_t n_in) {
    std::vector<const TensorInfo*> ts;
    size_t bytes = 0;
    int64_t rows = 0;
    for (const char* n : names) {
        const TensorInfo* t = tensor(il, n);
        if (!t) throw std::runtime_error("the GGUF has no tensor " + blk(file_layer(il), n));
        if (!ts.empty() && t->type != ts.front()->type)
            throw std::runtime_error(t->name + ": the matrices to stack differ in type");
        if ((int64_t) t->shape[0] != n_in) throw std::runtime_error(t->name + " is " + shape_of(*t) + ", not " +
                                                                  std::to_string(n_in) + " values a row");
        if (t->type != kTypeF32 && !kernels::native_mmvq_supported((int) t->type))
            throw std::runtime_error(t->name + ": a type the stacked products do not take");
        ts.push_back(t);
        bytes += home(*t).payload(*t);
        rows += (int64_t) (t->elements() / t->shape[0]);
    }
    const Dev d{dense_ ? dense_ + dense_at_ : nullptr, ts.front()->type, (int) n_in, (int) rows};
    dense_at_ += pad(bytes) + kTail;
    if (!dense_) return d;
    if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + ts.front()->name);
    size_t at = 0;
    for (const TensorInfo* t : ts) {
        const size_t b = home(*t).payload(*t);
        uploaded_.emplace_back(home(*t).data(*t), b);
        h2d((uint8_t*) d.p + at, home(*t).data(*t), b);
        at += b;
    }
    return d;
}

// The loop's message queries (loop_res_query, loop_kv_res_query: two rows a run of the core each) as one table, a
// later pass's four rows together - (q, experts) then (k, v) - so that one mix of the bank takes all four.
Dev Inkling2::msg_queries() {
    const TensorInfo* tq = gg_->find("loop_res_query.weight");
    const TensorInfo* tkv = gg_->find("loop_kv_res_query.weight");
    const int64_t rows = 2 * (core_end_ - core_start_);
    for (const TensorInfo* t : {tq, tkv})
        if (!t || t->type != kTypeF32 || (int64_t) t->shape[0] != n_embd_ || (int64_t) (t->elements() / n_embd_) != rows)
            throw std::runtime_error("unsupported model: loop_res_query / loop_kv_res_query are not F32 [n_embd, " +
                                     std::to_string(rows) + "]");
    const size_t row = (size_t) n_embd_ * sizeof(float), bytes = 2 * (size_t) rows * row;
    const Dev d{dense_ ? dense_ + dense_at_ : nullptr, kTypeF32, n_embd_, (int) (2 * rows)};
    dense_at_ += pad(bytes) + kTail;
    if (!dense_) return d;
    if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for the loop's message queries");
    const uint8_t* q = (const uint8_t*) home(*tq).data(*tq);
    const uint8_t* kv = (const uint8_t*) home(*tkv).data(*tkv);
    uploaded_.emplace_back(q, home(*tq).payload(*tq));
    uploaded_.emplace_back(kv, home(*tkv).payload(*tkv));
    std::vector<uint8_t> h(bytes);
    for (int64_t p = 0; p < rows; p += 2) {
        uint8_t* dst = h.data() + (size_t) (2 * p) * row;
        std::memcpy(dst, q + (size_t) p * row, 2 * row);
        std::memcpy(dst + 2 * row, kv + (size_t) p * row, 2 * row);
    }
    h2d(d.p, h.data(), bytes);
    return d;
}

Inkling2::~Inkling2() {
    if (head_host_) cudaFreeHost(head_host_);
    if (ds_.w1_host) cudaFreeHost(ds_.w1_host);
    if (igpu_) set_igpu(nullptr);
}

// INK_IGPU=1: the NVFP4 experts copied into memory the integrated GPU reads in place (Intel USM host memory), so that
// it can take some of a token's CPU experts while the CPU's threads compute the others (igpu.hpp).  RAM holds one copy
// still: the file's pages of them leave the process.  The Q8_0 layers (20-23) stay where they are, on the CPU.
void Inkling2::after_load() {
    if (ds_.on || std::getenv("INK_VRAM_REPORT"))
        std::fprintf(stderr, "strata-poly: DSpark drafter's VRAM: %.0f MiB of weights, %.0f MiB of scratch, %.0f MiB of "
                             "caches; the dense arena %.0f MiB, the scratch %.0f MiB in all\n",
                     ds_.vram_weights / 1048576.0, ds_.vram_scratch / 1048576.0, ds_.vram_own / 1048576.0,
                     dense_bytes_ / 1048576.0, scratch_at_ * 4.0 / 1048576.0);
    const char* on = std::getenv("INK_IGPU");
    if (!on || std::atoi(on) != 1 || igpu_) return;
    const Act a = act();
    igpu_ = Igpu::open(n_embd_, n_ff_exp_, a.beta, a.lb);
    if (!igpu_) return;
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const auto let_go = [page](const uint8_t* p, size_t n) {
        const uintptr_t lo = ((uintptr_t) p + page - 1) / page * page, hi = ((uintptr_t) p + n) / page * page;
        if (hi > lo) madvise((void*) lo, hi - lo, MADV_DONTNEED);
    };
    // copied by a few threads: one memcpy reads the file's cache at ~8 GB/s
    const auto copy = [](uint8_t* dst, const uint8_t* src, size_t n) {
        constexpr int kThreads = 8;
        std::vector<std::thread> th;
        const size_t part = (n / kThreads + 4095) / 4096 * 4096;
        for (int t = 0; t < kThreads; ++t) {
            const size_t at = std::min(n, (size_t) t * part), len = std::min(part, n - at);
            if (len) th.emplace_back([=] { std::memcpy(dst + at, src + at, len); });
        }
        for (std::thread& t : th) t.join();
    };
    size_t bytes = 0;
    int layers = 0;
    for (Layer& L : layers_) {
        if (!L.moe || L.gu_type != GGML_TYPE_NVFP4 || L.d_type != GGML_TYPE_NVFP4 || L.mixed() ||
            L.gu_stride != 2 * L.gu_bytes || L.exp_up != L.exp_gate + L.gu_bytes)
            continue;
        const size_t gu = L.gu_stride * (size_t) n_expert_, dn = L.d_bytes * (size_t) n_expert_;
        uint8_t* g = (uint8_t*) igpu_->alloc_host(gu);
        uint8_t* d = g ? (uint8_t*) igpu_->alloc_host(dn) : nullptr;
        if (!g || !d) {
            std::fprintf(stderr, "strata-poly: INK_IGPU: no more shared memory after %d layers\n", layers);
            break;
        }
        copy(g, L.exp_gate, gu);
        copy(d, L.exp_down, dn);
        let_go(L.exp_gate, gu);
        let_go(L.exp_down, dn);
        L.exp_gate = g;
        L.exp_up = g + L.gu_bytes;
        L.exp_down = d;
        bytes += gu + dn;
        ++layers;
    }
    for (Layer& L : layers_)   // a loop's later runs read their first run's (Engine::ex()); kept the same anyway
        if (L.owner >= 0) {
            const Layer& O = layers_[(size_t) L.owner];
            L.exp_gate = O.exp_gate;
            L.exp_up = O.exp_up;
            L.exp_down = O.exp_down;
        }
    if (layers == 0) {
        igpu_.reset();
        return;
    }
    // the RAM budget would count the copy as memory taken, and keep the experts in a pool of its own: map them all
    if (opt_.ram_experts_mib < -1) opt_.ram_experts_mib = -1;
    set_igpu(igpu_.get());
    std::fprintf(stderr, "strata-poly: INK_IGPU: %d layers' experts (%.1f GiB) where the iGPU reads them\n", layers,
                 (double) bytes / (1 << 30));
}

void Inkling2::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    plan();
}

void Inkling2::load_geometry() {
    // --draft-model is a model of its own (DSpark), not a file of this one with a block in it, which is what
    // load_shape() would look for
    std::unique_ptr<Shards> drafter = std::move(dg_);
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };
    dim_ = shape_i("attention.key_length");
    need(shape_i("attention.value_length") == dim_, "the attention's keys and values differ in length");
    const MetaValue* rot = find("rope.dimension_count");
    need(!rot || rot->u == 0, "rotary positions");
    window_ = shape_i("attention.sliding_window");
    n_rel_ = shape_i("attention.relative_length");
    extent_ = shape_i("attention.relative_extent");
    qk_clip_ = shape_f("attention.qk_clip");
    rel_clip_ = shape_f("attention.relative_clip");
    log_floor_ = (float) shape_i("attention.log_scale_floor");
    log_alpha_ = shape_f("attention.log_scale_alpha");
    taps_ = shape_i("shortconv.l_cache");
    block_size_ = shape_i("attn_res.block_size");
    n_unpadded_ = shape_i("vocab_size_unpadded");
    const MetaValue* cap = find("final_logit_softcapping");
    softcap_ = cap ? (float) cap->f : 0.f;
    const MetaValue* ws = find("expert_weights_scale");
    w_scale_ = ws ? (float) ws->num() : 0.f;
    situ_.kind = Act::kSitu;
    situ_.beta = shape_f("activation.situ_beta");
    situ_.lb = shape_f("activation.situ_linear_beta");
    const MetaValue* loops = find("num_loops");
    passes_ = loops ? (int) loops->u : 1;
    if (passes_ > 1) {
        core_start_ = shape_i("loop.core_start");
        core_end_ = shape_i("loop.core_end");
        gate_rank_ = shape_i("loop.gate_rank");
    }
    n_file_ = n_layer_ - (passes_ > 1 ? (passes_ - 1) * (core_end_ - core_start_) : 0);

    const MetaValue& heads = must("attention.head_count");
    const MetaValue& kv = must("attention.head_count_kv");
    const MetaValue& pattern = must("attention.sliding_window_pattern");
    n_head_ = (int) of_layer(heads, 0);
    n_kv_ = (int) of_layer(kv, 0);
    swa_.resize((size_t) n_layer_);
    for (int il = 0; il < n_layer_; ++il) {
        need(of_layer(heads, il) == n_head_ && of_layer(kv, il) == n_kv_, "layers with different heads");
        swa_[(size_t) il] = of_layer(pattern, il) != 0;
    }
    need(n_head_ > 0 && n_kv_ > 0 && n_head_ % n_kv_ == 0 && n_head_ / n_kv_ <= 4, "bad attention.head_count(_kv)");
    need(dim_ == 128, "heads of other than 128 values");
    need(n_rel_ > 0 && n_rel_ <= 16 && extent_ > 0 && window_ > 0, "bad relative bias or window");
    need(taps_ >= 2 && taps_ <= 8, "bad shortconv.l_cache");
    need(block_size_ > 0, "bad attn_res.block_size");
    need(n_unpadded_ > 0 && n_unpadded_ <= n_vocab_, "bad vocab_size_unpadded");
    need(passes_ < 2 || (core_start_ > 0 && core_start_ % block_size_ == 0 && core_end_ > core_start_ &&
                         (core_end_ - core_start_) % block_size_ == 0 && core_end_ <= n_file_ && gate_rank_ > 0),
         "the loop's core must start after the first block and span whole blocks");
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
    // windows of guessed tokens (--lookup): 8 by default.  Its many CUDA recordings outgrow VRAM in a long conversation;
    // Engine::record() then stops recording and the rest launch as they go.  8 rewrote a file 15% faster than 4.
    if (opt_.window == 0) opt_.window = 8;

    // the depth's streams: the stream, a slot for each block, and the bank a later pass reads
    int next = 1;
    block_slot_.assign((size_t) n_file_, -1);
    for (int p = 0; p < n_file_; p += block_size_) block_slot_[(size_t) p] = next++;
    bank_.clear();
    bank_states_ = InkStates{};
    if (passes_ > 1) {
        for (int p = core_start_ + block_size_; p < core_end_; p += block_size_) {
            bank_.push_back({block_slot_[(size_t) p], next});
            bank_states_.off[bank_states_.n++] = next++ * n_embd_;
        }
        bank_last_ = next++;
        bank_states_.off[bank_states_.n++] = bank_last_ * n_embd_;
        prelude_ = block_slot_[(size_t) core_start_];
    }
    n_streams_ = next;
    state_per_ = (size_t) (taps_ - 1) * (2 * (size_t) n_kv_ * dim_ + 2 * (size_t) n_embd_);
    if (drafter) {
        dg_ = std::move(drafter);
        block_ = dg_.get();   // the file its tensors are read from (home())
        ds_geometry();
    }
}

// What each run does with the streams, as llama.cpp's graph keeps its list of blocks: a run reads the blocks so far
// and the stream; the first run of a block keeps the stream before its attention as a block; a later pass starts
// from the blocks before the loop and the stream at its start.
void Inkling2::plan() {
    if (blocks_.empty()) return;
    auto with_cur = [&](const std::vector<int>& slots) {
        InkStates s{};
        for (int k : slots) s.off[s.n++] = k * n_embd_;
        s.off[s.n++] = cur_ * n_embd_;
        if (s.n > 8) throw std::runtime_error("unsupported model: more than 7 blocks of depth");
        return s;
    };
    std::vector<int> list, prelude_list;
    const int core = core_end_ - core_start_;
    for (int il = 0; il < n_layer_; ++il) {
        Block& B = blocks_[(size_t) il];
        const int ip = file_layer(il);
        B.file = ip;
        B.swa = swa_[(size_t) il] != 0;
        const bool in_core = passes_ > 1 && ip >= core_start_ && ip < core_end_ && il < core_start_ + passes_ * core;
        B.pass = in_core ? (il - core_start_) / core : 0;
        B.twin = in_core && B.pass > 0;
        if (in_core && ip == core_start_) {
            if (B.pass == 0) {
                prelude_list = list;
            } else {
                B.restart = true;
                list = prelude_list;
            }
        }
        if (B.pass > 0) B.msg_row = 2 * (ip - core_start_);
        B.attn_mix = ip > 0 ? with_cur(list) : InkStates{};
        if (ip % block_size_ == 0) {
            B.push = block_slot_[(size_t) ip];
            list.push_back(B.push);
        }
        B.ffn_mix = with_cur(list);
    }
    out_mix_ = with_cur(list);
}

void Inkling2::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    if (il == 0) {   // the model's own: the embedding's norm, the output's query, the loop's message queries
        const char* hs = std::getenv("INK_HEAD");
        const TensorInfo* head = gg_->find("output.weight");
        // the shortlist rescores from a Q8_0 matrix, or a Q6_K one (Unsloth's IQ4_XS file)
        head_short_ = !(hs && std::atoi(hs) == 0) && head && ink_rescore_supported((int) head->type) &&
                      (head->type != GGML_TYPE_Q6_K || n_embd_ % 256 == 0);
        if (head_short_) {
            out_q4_ = upload(head, "output.weight", n_embd_, n_vocab_, kGuessMat);
            // the Q8_0 matrix in pinned host memory, once: only the shortlist's rows are read, over the bus
            if (dense_ && !head_host_) {
                const size_t bytes = home(*head).payload(*head);
                ck(cudaHostAlloc(&head_host_, bytes, cudaHostAllocMapped), "the output matrix in host memory");
                std::memcpy(head_host_, home(*head).data(*head), bytes);
                void* dev = nullptr;
                ck(cudaHostGetDevicePointer(&dev, head_host_, 0), "the output matrix's card address");
                head_dev_ = dev;
            }
        }
        if (ds_.on) ds_load();
        tok_norm_ = upload(gg_->find("token_embd_norm.weight"), "token_embd_norm.weight", n_embd_, 1, kF32);
        out_query_ = upload(gg_->find("output_res_query.weight"), "output_res_query.weight", n_embd_, 1, kF32);
        if (passes_ > 1) msg_query_ = msg_queries();
    }
    if (B.twin) {
        // a later pass of the loop: its first run's weights, and its own caches, state and plan
        const Block& O = blocks_[(size_t) B.file];
        Block mine = B;
        B = O;
        B.swa = mine.swa;
        B.file = mine.file;
        B.pass = mine.pass;
        B.twin = true;
        B.attn_mix = mine.attn_mix;
        B.ffn_mix = mine.ffn_mix;
        B.push = mine.push;
        B.restart = mine.restart;
        B.msg_row = mine.msg_row;
        B.keys = B.vals = nullptr;
        B.state = nullptr;
        // the first run's norms and router; its routed experts and their cache are the first run's (Engine::ex())
        L = layers_[(size_t) B.file];
        L.moe = false;
        L.owner = B.file;
        L.slot.clear();
        return;
    }
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(B.file, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(B.file, n), a, b, kAnyMat); };
    L.fixed = true;   // the launches read the position on the card when they are recorded
    const int hq = n_head_ * dim_, kvw = n_kv_ * dim_;
    const TensorInfo* tq = tensor(il, "attn_q.weight");
    const TensorInfo* tg = tensor(il, "attn_gate.weight");
    const TensorInfo* tr = tensor(il, "attn_rel.weight");
    const TensorInfo* tk = tensor(il, "attn_k.weight");
    const TensorInfo* tv = tensor(il, "attn_v.weight");
    const bool same = tq && tg && tr && tq->type == tg->type && tq->type == tr->type;
    if (same && B.swa && tk && tv && tk->type == tq->type && tv->type == tq->type) {
        // a window layer: all five stacked; the others are slices of it (rows of the same type, back to back)
        B.wall = fused(il, {"attn_q.weight", "attn_gate.weight", "attn_rel.weight", "attn_k.weight", "attn_v.weight"},
                       n_embd_);
        const int qgr = 2 * hq + n_rel_ * n_head_;
        if (B.wall.n_out != qgr + 2 * kvw) throw std::runtime_error(blk(B.file, "attn_q/gate/rel/k/v: bad rows"));
        const size_t row = ggml_row_size((ggml_type) B.wall.type, n_embd_);
        auto slice = [&](int first, int rows) {
            Dev d = B.wall;
            d.p = B.wall.p ? (uint8_t*) B.wall.p + (size_t) first * row : nullptr;
            d.n_out = rows;
            return d;
        };
        B.wqgr = slice(0, qgr);
        B.wk = slice(qgr, kvw);
        B.wv = slice(qgr + kvw, kvw);
    } else if (same) {
        B.wqgr = fused(il, {"attn_q.weight", "attn_gate.weight", "attn_rel.weight"}, n_embd_);
        if (B.wqgr.n_out != 2 * hq + n_rel_ * n_head_) throw std::runtime_error(blk(B.file, "attn_q/gate/rel: bad rows"));
    } else {   // quantized apart (Unsloth's IQ4_XS): three products into the same rows
        B.wq = mat("attn_q.weight", n_embd_, hq);
        B.wgate = mat("attn_gate.weight", n_embd_, hq);
        B.wrel = mat("attn_rel.weight", n_embd_, (int64_t) n_rel_ * n_head_);
    }
    if (!B.wall.n_out) {
        B.wk = mat("attn_k.weight", n_embd_, kvw);
        B.wv = mat("attn_v.weight", n_embd_, kvw);
    }
    B.wo = mat("attn_output.weight", hq, n_embd_);
    B.wsink = mat("attn_sink_proj.weight", n_embd_, n_head_);
    B.q_clip = f32("attn_q_clip.weight", dim_, n_head_);
    B.rel_clip = f32("attn_rel_clip.weight", n_rel_, n_head_);
    B.rel_table = f32("attn_rel_bias.weight", n_rel_, B.swa ? window_ : extent_);
    B.k_conv = f32("attn_k_conv.weight", taps_, kvw);
    B.v_conv = f32("attn_v_conv.weight", taps_, kvw);
    B.attn_conv = f32("attn_post_conv.weight", taps_, n_embd_);
    B.ffn_conv = f32("ffn_post_conv.weight", taps_, n_embd_);
    if (B.swa) {
        L.attn_norm = f32("attn_norm.weight", n_embd_);
    } else {
        B.q_norm = f32("attn_q_inp_norm.weight", n_embd_);
        B.k_norm = f32("attn_k_inp_norm.weight", n_embd_);
        B.v_norm = f32("attn_v_inp_norm.weight", n_embd_);
        B.kconv_proj = mat("attn_k_conv_proj.weight", n_embd_, (int64_t) taps_ * n_kv_);
        B.vconv_proj = mat("attn_v_conv_proj.weight", n_embd_, (int64_t) taps_ * n_kv_);
    }
    if (B.file > 0) B.attn_query = f32("attn_res_query.weight", n_embd_, B.swa ? 1 : 3);
    B.ffn_query = f32("ffn_res_query.weight", n_embd_, 1);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    if (passes_ > 1 && B.file >= core_start_ && B.file < core_end_) {
        static const char* const kModule[4] = {"attn", "k", "v", "mlp"};
        for (int m = 0; m < 4; ++m) {
            const std::string d = std::string("loop_") + kModule[m] + "_down.weight";
            const std::string g = std::string("loop_") + kModule[m] + "_gain.weight";
            const std::string f = std::string("loop_") + kModule[m] + "_forget.weight";
            B.loop_down[m] = mat(d.c_str(), 2 * (int64_t) n_embd_, gate_rank_);
            B.loop_gf[m] = fused(il, {g.c_str(), f.c_str()}, gate_rank_);
            if (B.loop_gf[m].n_out != 2 * n_embd_) throw std::runtime_error(blk(B.file, g.c_str()) + ": bad shape");
        }
    }
    load_moe(il, false);
    const TensorInfo* b = tensor(il, "exp_probs_b.bias");
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (b) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(blk(B.file, "exp_probs_b.bias") + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(B.file, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
}

size_t Inkling2::recurrent_floats() const { return (size_t) n_layer_ * state_per_; }

// The widest input lin() takes: a loop controller's two normed rows, or the attention's heads.
int Inkling2::widest() const { return std::max({n_head_ * dim_, 2 * n_embd_, ds_.on ? ds_.n_ff : 0}); }

// A full run keeps a row a position; a window run a ring of the rows a call can read.  Each run's convolutions keep
// their past inputs in state_.
void Inkling2::alloc_caches() {
    // INK_L2=1: the first loop layers' weights (read three times a token) in the persisting part of L2, as much as the
    // card lets one window take
    static bool l2_done = false;
    const char* l2 = std::getenv("INK_L2");
    if (!l2_done && l2 && std::atoi(l2) == 1 && passes_ > 1 && dense_) {
        l2_done = true;
        int most = 0, win = 0;
        cudaDeviceGetAttribute(&most, cudaDevAttrMaxPersistingL2CacheSize, device_);
        cudaDeviceGetAttribute(&win, cudaDevAttrMaxAccessPolicyWindowSize, device_);
        const Block& C = blocks_[(size_t) core_start_];
        const uint8_t* from = (const uint8_t*) (C.wall.p ? C.wall.p : C.wqgr.p);
        const size_t bytes = std::min<size_t>((size_t) win, dense_ + dense_bytes_ - from);
        if (most > 0 && from && cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t) most) == cudaSuccess) {
            cudaStreamAttrValue a{};
            a.accessPolicyWindow.base_ptr = (void*) from;
            a.accessPolicyWindow.num_bytes = bytes;
            a.accessPolicyWindow.hitRatio = std::min(1.f, (float) most / (float) bytes);
            a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            const cudaError_t e = cudaStreamSetAttribute(cs(st_), cudaStreamAttributeAccessPolicyWindow, &a);
            std::fprintf(stderr, "strata-poly: L2 keeps %d MiB of %zu MiB of the loop's weights%s\n", most >> 20, bytes >> 20,
                         e == cudaSuccess ? "" : " (refused)");
        }
        cudaGetLastError();
    }
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    const size_t row = (size_t) n_kv_ * dim_;
    if (q8()) {   // a row: its codes (store_q8's, or ink_store_q4's at half the bytes), and in an array of their own
                  // its scales
        const size_t kcode = v4() == 2 ? row / 2 : row, vcode = v4() ? row / 2 : row;
        const size_t scale = row / 32 * sizeof(uint16_t), per = kcode + vcode + 2 * scale;
        size_t rings = 0;
        for (const Block& B : blocks_)
            if (B.swa) rings += (size_t) ring_ * per;
        uint8_t* kv = rings ? (uint8_t*) seq_alloc(rings, "the attention cache") : nullptr;
        for (int il = 0; il < (int) blocks_.size(); ++il) {
            Block& B = blocks_[(size_t) il];
            if (!runs(il)) continue;
            B.state = state_ + (size_t) il * state_per_;
            if (B.swa) {
                B.k_codes = (int8_t*) kv;
                kv += (size_t) ring_ * kcode;
                B.v_codes = (int8_t*) kv;
                kv += (size_t) ring_ * vcode;
                B.k_sc = (uint16_t*) kv;
                kv += (size_t) ring_ * scale;
                B.v_sc = (uint16_t*) kv;
                kv += (size_t) ring_ * scale;
            } else {
                B.k_codes = (int8_t*) kv_alloc(kcode, 1);
                B.v_codes = (int8_t*) kv_alloc(vcode, 1);
                B.k_sc = (uint16_t*) kv_alloc(scale, 1);
                B.v_sc = (uint16_t*) kv_alloc(scale, 1);
            }
        }
        return;
    }
    size_t rings = 0;
    for (const Block& B : blocks_)
        if (B.swa) rings += 2 * (size_t) ring_ * row;
    uint16_t* kv = rings ? (uint16_t*) seq_alloc(rings * sizeof(uint16_t), "the attention cache") : nullptr;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        B.state = state_ + (size_t) il * state_per_;
        if (B.swa) {
            B.keys = kv;
            kv += (size_t) ring_ * row;
            B.vals = kv;
            kv += (size_t) ring_ * row;
        } else {
            B.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }
}

void Inkling2::assign_scratch() {
    const size_t n = (size_t) rows_, e = (size_t) n_embd_, hq = (size_t) n_head_ * dim_, kvw = (size_t) n_kv_ * dim_;
    mix_ = fbuf(3 * n * e);
    msg_ = fbuf(4 * n * e);
    // a controller's rows, or the three controllers' of a call of a few tokens (control3): no more scratch than one
    // controller's of a prompt batch, which the expert cache would give up
    const size_t ctl_rows = std::max(n, 3 * (size_t) std::min(rows_, kStageRows - 1));
    ctl_ = fbuf(2 * ctl_rows * e);
    ctl_h_ = fbuf(ctl_rows * (size_t) std::max(gate_rank_, 1));
    ctl_g_ = fbuf(2 * n * e);   // gain then forget, a token's 2 * n_embd together
    xin_ = fbuf(3 * n * e);
    q_ = fbuf(n * (2 * hq + (size_t) n_head_ * n_rel_ + 2 * kvw));   // a token's q, gate, rel (and k, v: wall)
    sink_ = fbuf(n * (size_t) n_head_);
    kraw_ = fbuf(n * kvw);
    k_ = fbuf(n * kvw);
    vraw_ = fbuf(n * kvw);
    v_ = fbuf(n * kvw);
    dyn_ = fbuf(2 * n * (size_t) taps_ * n_kv_);   // the keys' taps, then the values
    att_ = fbuf(n * hq);
    part_ = fbuf(ink_attend_floats(rows_, n_head_, opt_.max_context));
    if (const size_t counters = ink_attend_counters(rows_, n_kv_, ds_.on)) {
        attn_counters_ = (unsigned int*) fbuf(counters);
        if (attn_counters_)
            ck(cudaMemsetAsync(attn_counters_, 0, counters * sizeof(*attn_counters_), cs(st_)),
               "attention completion counters");
    }
    tmp_ = fbuf(n * e);
    if (head_short_ && !head_ids_) {
        const size_t r = (size_t) std::max(opt_.window, 8);
        head_ids_ = (int*) dmalloc(r * kHeadCap * sizeof(int), "the output's shortlist");
        head_count_ = (int*) dmalloc(r * sizeof(int), "the output's shortlist");
        head_scratch_ = (float*) dmalloc(ink_shortlist_floats((int) r) * sizeof(float), "the output's shortlist");
        const char* hc = std::getenv("INK_HEAD_CHECK");
        if (hc && std::atoi(hc) == 1) head_exact_ = (float*) dmalloc(r * (size_t) n_vocab_ * sizeof(float), "head check");
    }
    // the attention's three inputs in Q8_1, then its gated output (fewer values a token than the three inputs), for a
    // call of fewer than kStageRows tokens (layer_head)
    xq3_ = fbuf(3 * (size_t) std::min(rows_, kStageRows - 1) * kernels::native_q8_1_bytes(n_embd_) / sizeof(float));
    if (!row_off_) row_off_ = (long long*) dmalloc(8 * sizeof(long long), "the step's state offsets");
    if (ds_.on) ds_scratch();
}

// The loop's message controller of module m (q, k, v, experts): x = (1 - tanh(F h)) x + tanh(G h) msg, where
// h = D [rms(x); rms(msg)].
void Inkling2::control(Block& B, int m, float* x, const float* msg, int n) {
    ink_rms2(x, msg, ctl_, n_embd_, n, kMixEps, st_);
    lin_f32(B.loop_down[m], ctl_, ctl_h_, n);
    // G h, F h and the gate in one launch, where the product would be ink_gemv_f32's (lin_f32)
    const Dev& gf = B.loop_gf[m];
    const float* w = fp(gf);
    const float* h = ctl_h_;
    const float* cur = x;
    if (gf.type == kTypeF32 && gf.scale == 1.0f && n < kStageRows &&
        ink_gate_gemv(&w, &h, &cur, n_embd_, &msg, n_embd_, &x, 1, gf.n_in, n_embd_, n, st_))
        return;
    lin_f32(gf, ctl_h_, ctl_g_, n);
    ink_gate(x, n_embd_, msg, n_embd_, ctl_g_, ctl_g_ + n_embd_, 2 * n_embd_, x, n_embd_, n, st_);
}

// The attention's three controllers (q, k and v: modules 0 to 2), out[j] from x[j] and msg[j] as control() makes it
// in place: each step of the three in one launch where the products are ink_gemv_f32's (a few tokens), the same sums.
void Inkling2::control3(Block& B, float* const* x, const float* const* msg, float* const* out, int n) {
    const size_t e = (size_t) n_embd_, rank = (size_t) std::max(gate_rank_, 1);
    bool jobs = n < kStageRows;
    for (int j = 0; j < 3; ++j)
        for (const Dev* d : {&B.loop_down[j], &B.loop_gf[j]})
            jobs = jobs && d->type == kTypeF32 && d->scale == 1.0f;
    if (!jobs) {   // a prompt batch: control()'s launches each, into out[j]
        for (int j = 0; j < 3; ++j) {
            ink_rms2(x[j], msg[j], ctl_, n_embd_, n, kMixEps, st_);
            lin_f32(B.loop_down[j], ctl_, ctl_h_, n);
            lin_f32(B.loop_gf[j], ctl_h_, ctl_g_, n);
            ink_gate(x[j], n_embd_, msg[j], n_embd_, ctl_g_, ctl_g_ + n_embd_, 2 * n_embd_, out[j], n_embd_, n, st_);
        }
        return;
    }
    const float* xs[3] = {x[0], x[1], x[2]};
    const float* down[3];
    const float* gf[3];
    const float* c[3];
    const float* hc[3];
    float* h[3];
    for (int j = 0; j < 3; ++j) {
        down[j] = fp(B.loop_down[j]);
        gf[j] = fp(B.loop_gf[j]);
        c[j] = ctl_ + (size_t) j * n * 2 * e;
        h[j] = ctl_h_ + (size_t) j * n * rank;
        hc[j] = h[j];
    }
    ink_rms2_jobs(xs, msg, 3, ctl_, n_embd_, n, kMixEps, st_);
    if (!ink_gemv_f32_jobs(down, c, h, 3, B.loop_down[0].n_in, B.loop_down[0].n_out, n, st_))
        for (int j = 0; j < 3; ++j) lin_f32(B.loop_down[j], c[j], h[j], n);
    if (ink_gate_gemv(gf, hc, xs, n_embd_, msg, n_embd_, out, 3, B.loop_gf[0].n_in, n_embd_, n, st_)) return;
    for (int j = 0; j < 3; ++j) {
        lin_f32(B.loop_gf[j], h[j], ctl_g_, n);
        ink_gate(x[j], n_embd_, msg[j], n_embd_, ctl_g_, ctl_g_ + n_embd_, 2 * n_embd_, out[j], n_embd_, n, st_);
    }
}

// A run from its input to its router's copy.  r_ holds a token's streams; slot 0 is the residual.
void Inkling2::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int rs = n_streams_ * n_embd_;
    const size_t plane = (size_t) n * n_embd_;
    float* cur = r_ + (size_t) cur_ * n_embd_;
    if (rows_of_) {   // step(): where each row's sequence keeps its state, from the bound one's (its tail reads it too)
        long long off[8];
        for (int i = 0; i < n; ++i) off[i] = (long long) (seq_state(rows_of_[i].seq, state_) - state_);
        ink_set_offsets(row_off_, off, n, st_);
    }
    if (il == 0 && n == 1 && (images_.empty() || rows_of_ || !image_at(pos_))) {   // a token: normed where it is
        rms_norm(cur, fp(tok_norm_), cur, n_embd_, 1, rms_eps_, st_);
    } else if (il == 0) {   // the token's embedding, normed; a picture's rows go in as the encoder made them (llama.cpp
                     // norms only tokens).  Row t is at pos_ + t here: a batch with a picture's rows is not replayed
                     // from a recording, and step()'s rows (rows_of_) are tokens an answer writes
        ink_rows(cur, rs, tmp_, n_embd_, n_embd_, n, false, st_);
        rms_norm(tmp_, fp(tok_norm_), tmp_, n_embd_, n, rms_eps_, st_);
        if (images_.empty() || rows_of_) {
            ink_rows(tmp_, n_embd_, cur, rs, n_embd_, n, false, st_);
        } else {
            for (int a = 0; a < n;) {   // the normed rows back for each run of text rows
                if (image_at(pos_ + a)) {
                    ++a;
                    continue;
                }
                int b = a;
                while (b < n && !image_at(pos_ + b)) ++b;
                ink_rows(tmp_ + (size_t) a * n_embd_, n_embd_, cur + (size_t) a * rs, rs, n_embd_, b - a, false, st_);
                a = b;
            }
        }
    }
    if (B.restart) {   // a later pass: the pass before's blocks to the bank, and the stream from the loop's start
        RowCopies c{};
        int k = 0;
        const auto copy = [&](int from, int to) {
            if (k == 8) {
                ink_copies(c, k, rs, n_embd_, n, st_);
                k = 0;
            }
            c.src[k] = r_ + (size_t) from * n_embd_;
            c.dst[k++] = r_ + (size_t) to * n_embd_;
        };
        for (const auto& [from, to] : bank_) copy(from, to);
        copy(cur_, bank_last_);
        copy(prelude_, cur_);
        ink_copies(c, k, rs, n_embd_, n, st_);
    }
    if (ds_.on) ds_capture(il, cur, rs, n);   // DSpark: the run's input, if it is one of the drafter's
    const bool msgs = B.pass > 0;
    // the messages, four planes of msg_: (q, experts), then (k, v) - one mix of the bank with four queries
    float* const msg_b = msg_ + 2 * plane;
    if (msgs) ink_mix(r_, rs, bank_states_, fp(msg_query_) + (size_t) 2 * B.msg_row * n_embd_, 4, msg_, plane, n_embd_, n, st_);
    // the attention's inputs: a window run reads one mix of the depth, a full run one each for q, k and v
    float* in[3];
    const int nq = B.swa ? 1 : 3;
    if (B.attn_mix.n > 0) ink_mix(r_, rs, B.attn_mix, fp(B.attn_query), nq, mix_, plane, n_embd_, n, st_);
    else ink_rows(cur, rs, mix_, n_embd_, n_embd_, n, false, st_);
    for (int j = 0; j < 3; ++j) in[j] = mix_ + (size_t) (B.attn_mix.n > 0 && !B.swa ? j : 0) * plane;
    if (msgs) {   // the messages differ by input: three inputs, from the one mix of a window run, into xin_
        const float* m[3] = {msg_, msg_b, msg_b + plane};
        float* out[3];
        for (int j = 0; j < 3; ++j) out[j] = xin_ + (size_t) j * plane;
        control3(B, in, m, out, n);
        for (int j = 0; j < 3; ++j) in[j] = out[j];   // their norms below are in place
    }
    float* xq = xin_;
    float* xk = xin_ + plane;
    float* xv = xin_ + 2 * plane;
    // the inputs' norms (a window run's one input, or q, k and v) in one launch, which for a few tokens also
    // quantizes them for the products: the bytes lin() would make of them
    const bool one = B.swa && !msgs;
    const int planes = one ? 1 : 3;
    const float* nw[3];
    for (int j = 0; j < 3; ++j) nw[j] = fp(B.swa ? L.attn_norm : j == 0 ? B.q_norm : j == 1 ? B.k_norm : B.v_norm);
    const void* xq8[3] = {nullptr, nullptr, nullptr};
    if (n < kStageRows) {
        kernels::native_rms_norm_q8_1_planes(in, nw, planes, xin_, xq3_, n_embd_, n, rms_eps_, sum_threads(n_embd_), st_);
        const size_t qp = (size_t) n * kernels::native_q8_1_bytes(n_embd_);
        for (int j = 0; j < 3; ++j) xq8[j] = (const uint8_t*) xq3_ + (size_t) (one ? 0 : j) * qp;
    } else {
        for (int j = 0; j < planes; ++j) rms_norm(in[j], nw[j], xin_ + (size_t) j * plane, n_embd_, n, rms_eps_, st_);
    }
    if (one) xk = xv = xq;
    dbg("attn_in", il, xq, n_embd_);

    // self-attention
    {
        const long long* at = recording_ ? at_ : nullptr;
        const int hq = n_head_ * dim_, kvw = n_kv_ * dim_;
        const size_t keep = (size_t) (taps_ - 1);
        const float conv_s = 1.f / std::sqrt((float) (taps_ + 1)), conv_d = 1.f / std::sqrt((float) (2 * taps_ + 1));
        // one product for q, gate, rel, k and v where they read one input; a token's row is then all five
        const bool all = B.wall.n_out > 0 && xk == xq && xv == xq;
        const int qgr = all ? B.wall.n_out : 2 * hq + n_head_ * n_rel_;   // floats of a token's q, gate and rel row
        float* gate = q_ + hq;
        float* rel = q_ + 2 * hq;
        if (all) {
            lin(B.wall, xq, q_, n, xq8[0]);
        } else if (B.wqgr.n_out > 0) {
            lin(B.wqgr, xq, q_, n, xq8[0]);
        } else {
            const Dev* part[3] = {&B.wq, &B.wgate, &B.wrel};
            const int at_col[3] = {0, hq, 2 * hq};
            for (int j = 0; j < 3; ++j) {   // att_ is free until the attention writes it
                lin(*part[j], xq, att_, n, xq8[0]);
                ink_rows(att_, part[j]->n_out, q_ + at_col[j], qgr, part[j]->n_out, n, false, st_);
            }
        }
        lin(B.wsink, xq, sink_, n);
        const float* kin = all ? q_ + (2 * hq + n_head_ * n_rel_) : kraw_;
        const float* vin = all ? kin + kvw : vraw_;
        const int kv_stride = all ? qgr : kvw;
        const bool rows = rows_of_ != nullptr;   // step(): each row a sequence of its own, with its own state
        // INK_KV_SIM=v4|v4r|k4v4r|k4r (a quality experiment, with --no-graphs): the rows a 4-bit cache would give
        // back, unrotated or rotated as llama.cpp rotates them (V in blocks of 64, K by the head), into the Q8_0 store
        static const std::string kv_sim = std::getenv("INK_KV_SIM") ? std::getenv("INK_KV_SIM") : "";
        // a token's keys and values from their products into a Q8_0 or k8v4 cache in one launch (ink_kv_chain): their
        // convolutions, the keys' clip, the values' norm and the stores
        static const bool enable_v4_chain = [] {
            const char* value = std::getenv("INK_KV_CHAIN_V4");
            return value && std::strcmp(value, "1") == 0;
        }();
        const bool chain_v4 = v4() == 1 && enable_v4_chain;
        const bool chain = n == 1 && !rows && q8() && (v4() == 0 || chain_v4) && kv_sim.empty();
        const float alpha = B.swa ? 0.f : log_alpha_;   // the queries' and relative terms' position factor
        float* const dyn_v = dyn_ + (size_t) n * taps_ * n_kv_;   // the values' taps beside the keys'
        if (!all) lin(B.wk, xk, kraw_, n, xq8[1]);
        if (!B.swa) lin(B.kconv_proj, xk, dyn_, n);
        if (rows) {
            ink_conv_rows(kin, kv_stride, B.state, row_off_, fp(B.k_conv), B.swa ? nullptr : dyn_, n_kv_, k_, kvw,
                          taps_, n, B.swa ? conv_s : conv_d, st_);
        } else if (!chain) {
            ink_conv(kin, B.state, fp(B.k_conv), B.swa ? nullptr : dyn_, n_kv_, k_, kvw, taps_, n,
                     B.swa ? conv_s : conv_d, st_, snap_bits_ ? snap_at_ + (B.state - state_) : nullptr, rec_n_,
                     snap_bits_, kv_stride);
        }
        if (!all) lin(B.wv, xv, vraw_, n, xq8[2]);
        if (!B.swa) lin(B.vconv_proj, xv, dyn_v, n);
        if (chain) {
            const float* kdyn = B.swa ? nullptr : dyn_;
            const float* vdyn = B.swa ? nullptr : dyn_v;
            float* ks = snap_bits_ & 1u ? snap_at_ + (B.state - state_) : nullptr;
            float* vs = snap_bits_ & 1u ? snap_at_ + (B.state + keep * kvw - state_) : nullptr;
            const InkKvSide ksd{kin, B.state, fp(B.k_conv), kdyn, k_, k_, B.k_codes, B.k_sc, ks};
            const InkKvSide vsd{vin, B.state + keep * kvw, fp(B.v_conv), vdyn, v_, vraw_, B.v_codes, B.v_sc, vs};
            const InkClip cq{q_, fp(B.q_clip), qk_clip_, alpha, dim_, n_head_};
            const InkClip cr{rel, fp(B.rel_clip), rel_clip_, alpha, n_rel_, n_head_};
            ink_kv_chain(ksd, vsd, n_kv_, dim_, taps_, B.swa ? conv_s : conv_d, qk_clip_, rms_eps_, pos_, at,
                         B.swa ? ring_ : 0, cq, cr, log_floor_, chain_v4, st_);
        } else if (rows) {
            ink_conv_rows(vin, kv_stride, B.state + keep * kvw, row_off_, fp(B.v_conv), B.swa ? nullptr : dyn_v,
                          n_kv_, v_, kvw, taps_, n, B.swa ? conv_s : conv_d, st_);
        } else {
            ink_conv(vin, B.state + keep * kvw, fp(B.v_conv), B.swa ? nullptr : dyn_v, n_kv_, v_, kvw, taps_, n,
                     B.swa ? conv_s : conv_d, st_,
                     snap_bits_ ? snap_at_ + (B.state + keep * kvw - state_) : nullptr, rec_n_, snap_bits_, kv_stride);
        }
        // q, k and the relative terms clipped (q and r scaled by the position on the full runs), v normed: the
        // chain's are already
        if (rows) {   // a row's position is its own: at_ holds one a row
            for (int i = 0; i < n; ++i)
                ink_clip3(q_ + (size_t) i * qgr, qgr, fp(B.q_clip), qk_clip_, dim_, n_head_, alpha, k_ + (size_t) i * kvw,
                          kvw, nullptr, qk_clip_, dim_, n_kv_, 0.f, rel + (size_t) i * qgr, qgr, fp(B.rel_clip),
                          rel_clip_, n_rel_, n_head_, alpha, 1, log_floor_, rows_of_[i].pos, at_ + 2 * i, st_);
        } else if (!chain) {
            ink_clip3(q_, qgr, fp(B.q_clip), qk_clip_, dim_, n_head_, alpha, k_, kvw, nullptr, qk_clip_, dim_, n_kv_, 0.f,
                      rel, qgr, fp(B.rel_clip), rel_clip_, n_rel_, n_head_, alpha, n, log_floor_, pos_, at, st_);
        }
        if (!chain) head_norm_rope(v_, dim_, nullptr, vraw_, dim_, n_kv_, 0, 1.f, pos_, n, rms_eps_, st_);
        const float* k = k_;
        const int ring = B.swa ? ring_ : 0;
        // INK_KV4_CHECK=1 (k8v4 or k4v4, with --no-graphs): the same attention over Q8_0 caches of the rows
        // INK_KV_SIM's 4-bit simulation gives, and of the rows themselves, to compare the real cache's output with
        static const bool kv4_check_env = std::getenv("INK_KV4_CHECK") != nullptr;
        const bool kv4_check = kv4_check_env && v4() && !rows && !at;
        // k4v4: the keys are rotated by the head, and so are the queries (q.k is the same)
        if (v4() == 2 && rows) ink_hadamard(q_, qgr, hq, n, dim_, st_);
        if (rows) {   // each row into its own sequence's caches at its own position, and its attention there
            for (int i = 0; i < n; ++i) {
                const Row& r = rows_of_[i];
                const long long* ai = at_ + 2 * i;
                const float* kr = k + (size_t) i * kvw;
                const float* vr = vraw_ + (size_t) i * kvw;
                const void* kc;
                const void* vc;
                const uint16_t* ks = nullptr;
                const uint16_t* vs = nullptr;
                if (q8()) {
                    int8_t* kcode = (int8_t*) seq_buf(r.seq, B.k_codes);
                    int8_t* vcode = (int8_t*) seq_buf(r.seq, B.v_codes);
                    uint16_t* ksc = (uint16_t*) seq_buf(r.seq, B.k_sc);
                    uint16_t* vsc = (uint16_t*) seq_buf(r.seq, B.v_sc);
                    if (v4() == 2) ink_store_q4(kr, (uint8_t*) kcode, ksc, kvw, 0, ai, 1, ring, dim_, st_);
                    else store_q8_at(kr, kcode, ksc, kvw, ai, 1, ring, st_);
                    if (v4()) ink_store_q4(vr, (uint8_t*) vcode, vsc, kvw, 0, ai, 1, ring, kV4Block, st_);
                    else store_q8_at(vr, vcode, vsc, kvw, ai, 1, ring, st_);
                    kc = kcode, vc = vcode, ks = ksc, vs = vsc;
                } else {
                    uint16_t* kcache = (uint16_t*) seq_buf(r.seq, B.keys);
                    uint16_t* vcache = (uint16_t*) seq_buf(r.seq, B.vals);
                    store_f16_at(kr, kcache, kvw, ai, 1, ring, st_);
                    store_f16_at(vr, vcache, kvw, ai, 1, ring, st_);
                    kc = kcache, vc = vcache;
                }
                const int64_t keep_pos = pos_;
                pos_ = r.pos;
                const int sized = B.swa ? std::min(window_, opt_.max_context) : room(1);
                pos_ = keep_pos;
                ink_attend(q_ + (size_t) i * qgr, qgr, kc, vc, rel + (size_t) i * qgr, qgr, fp(B.rel_table), n_rel_,
                           B.swa ? window_ : extent_, sink_ + (size_t) i * n_head_, att_ + (size_t) i * hq, part_, dim_,
                           n_head_, n_kv_, 1.f / std::sqrt((float) dim_), r.pos, 1, B.swa ? window_ : 0, ring, st_, ai,
                           sized, ks, vs, v4(), attn_counters_);
            }
        }
        static float* sim_buf = nullptr;
        static size_t sim_n = 0;
        for (int side = 0; side < 2 && q8() && !rows && !chain; ++side) {
            const float* x = side ? vraw_ : k;
            if (!kv_sim.empty() && !v4() && kv_sim.find(side ? "v4" : "k4") != std::string::npos) {
                const size_t need = (size_t) n * kvw;
                if (need > sim_n) {
                    if (sim_buf) cudaFree(sim_buf);
                    ck(cudaMalloc(&sim_buf, need * sizeof(float)), "INK_KV_SIM");
                    sim_n = need;
                }
                const bool rot = kv_sim.back() == 'r';
                ink_kv_sim(x, sim_buf, n, kvw, rot ? (side ? 64 : dim_) : 32, rot ? 1 : 0, st_);
                x = sim_buf;
            }
            int8_t* codes = side ? B.v_codes : B.k_codes;
            uint16_t* sc = side ? B.v_sc : B.k_sc;
            if (side ? v4() > 0 : v4() == 2) {   // 4-bit: rotated (the values in blocks of 64, the keys by the head)
                ink_store_q4(x, (uint8_t*) codes, sc, kvw, pos_, at, n, ring, side ? kV4Block : dim_, st_);
            } else if (at) {
                store_q8_at(x, codes, sc, kvw, at, n, ring, st_);
            } else if (!B.swa) {
                store_q8(x, codes, sc, kvw, pos_, n, st_);
            } else {
                const int a = (int) (pos_ % ring_), head = std::min(n, ring_ - a);
                store_q8(x, codes, sc, kvw, a, head, st_);
                if (head < n) store_q8(x + (size_t) head * kvw, codes, sc, kvw, 0, n - head, st_);
            }
        }
        for (int side = 0; side < 2 && !q8() && !rows; ++side) {
            const float* x = side ? vraw_ : k;
            uint16_t* cache = side ? B.vals : B.keys;
            if (at) {
                store_f16_at(x, cache, kvw, at, n, ring, st_);
            } else if (!B.swa) {
                store_f16(x, cache, kvw, pos_, n, st_);
            } else {
                const int a = (int) (pos_ % ring_), head = std::min(n, ring_ - a);
                store_f16(x, cache, kvw, a, head, st_);
                if (head < n) store_f16(x + (size_t) head * kvw, cache, kvw, 0, n - head, st_);
            }
        }
        if (kv4_check) kv4_shadow(B, il, n, k, vraw_, qgr, rel, ring);
        if (v4() == 2 && !rows) ink_hadamard(q_, qgr, hq, n, dim_, st_);
        // a few tokens' launches are sized as a recorded one's whether recorded or not (keep() puts tokens through
        // again unrecorded): a token's attention sums in the same order as in a call of its own
        const bool size_for_graph = at || (ds_.on && n <= 8);
        const int room_rows = size_for_graph ? (B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n)) : 0;
        if (!rows) ink_attend(q_, qgr, q8() ? (const void*) B.k_codes : B.keys, q8() ? (const void*) B.v_codes : B.vals, rel, qgr,
                   fp(B.rel_table), n_rel_, B.swa ? window_ : extent_, sink_, att_, part_, dim_, n_head_, n_kv_,
                   1.f / std::sqrt((float) dim_), pos_, n, B.swa ? window_ : 0, ring, st_, at, room_rows,
                   q8() ? B.k_sc : nullptr, q8() ? B.v_sc : nullptr, v4(), attn_counters_, ds_.on);
        if (kv4_check) kv4_compare(B, il, n);
        if (n < kStageRows) {   // the gate, and the Q8_1 blocks of the product's input, in one launch
            ink_mul_sigmoid_q8(att_, gate, hq, n, qgr, xq3_, st_);
            lin(B.wo, att_, tmp_, n, xq3_);
        } else {
            mul_sigmoid(att_, gate, hq, n, qgr, false, st_);
            lin(B.wo, att_, tmp_, n);
        }
        // into the residual: the first run of a block keeps the stream as a block and starts again from the
        // attention's output, the others add it
        float* push = B.push >= 0 ? r_ + (size_t) B.push * n_embd_ : nullptr;
        if (rows) {
            ink_conv_rows(tmp_, n_embd_, B.state + 2 * keep * kvw, row_off_, fp(B.attn_conv), nullptr, 1, ao_, n_embd_,
                          taps_, n, conv_s, st_);
            if (push) ink_rows(cur, rs, push, rs, n_embd_, n, false, st_);
            ink_rows(ao_, n_embd_, cur, rs, n_embd_, n, !push, st_);
        } else {
            ink_conv(tmp_, B.state + 2 * keep * kvw, fp(B.attn_conv), nullptr, 1, ao_, n_embd_, taps_, n, conv_s, st_,
                     snap_bits_ ? snap_at_ + (B.state + 2 * keep * kvw - state_) : nullptr, rec_n_, snap_bits_, 0, cur,
                     rs, push);
        }
    }
    dbg("attn", il, ao_, n_embd_);

    // the experts' input
    ink_mix(r_, rs, B.ffn_mix, fp(B.ffn_query), 1, mix_, plane, n_embd_, n, st_);
    if (msgs) control(B, 3, mix_, msg_ + plane, n);
    kernels::native_add_rms_norm_q8_1(mix_, nullptr, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_,
                                      sum_threads(n_embd_), st_);
    dbg("ffn_in", il, xn_, n_embd_);
    Layer& X = layers_[(size_t) ex(il)];   // the experts' layer: this one, or the loop's first run of it
    if (!batched(X, n)) moe_router(X, n);
}

// INK_KV4_CHECK: the run's attention over two Q8_0 caches beside the real 4-bit one - of the rows INK_KV_SIM's
// simulation gives for the 4-bit sides ([0]) and of the rows themselves ([1]); the keys where they are Q8_0 are the
// real cache's - into shadow_out_.  After the real cache's stores, before the queries are rotated.
void Inkling2::kv4_shadow(const Block& B, int il, int n, const float* k, const float* v, int qgr, const float* rel,
                          int ring) {
    const int kvw = n_kv_ * dim_, hq = n_head_ * dim_;
    if (shadow_.empty()) {
        shadow_.resize(blocks_.size());
        ck(cudaMalloc(&shadow_rows_, (size_t) rows_ * kvw * sizeof(float)), "INK_KV4_CHECK");
        ck(cudaMalloc(&shadow_out_, 2 * (size_t) rows_ * hq * sizeof(float)), "INK_KV4_CHECK");
    }
    const size_t cap = B.swa ? (size_t) ring_ : (size_t) opt_.max_context;
    for (Shadow& sh : shadow_[(size_t) il]) {
        if (sh.vc) continue;
        const auto get = [&](size_t bytes) {
            void* p = nullptr;
            ck(cudaMalloc(&p, cap * bytes), "INK_KV4_CHECK");
            return p;
        };
        sh.vc = (int8_t*) get(kvw);
        sh.vs = (uint16_t*) get(kvw / 32 * sizeof(uint16_t));
        if (v4() == 2) {
            sh.kc = (int8_t*) get(kvw);
            sh.ks = (uint16_t*) get(kvw / 32 * sizeof(uint16_t));
        }
    }
    for (int j = 0; j < 2; ++j) {
        Shadow& sh = shadow_[(size_t) il][(size_t) j];
        for (int side = 0; side < 2; ++side) {
            if (side == 0 && v4() != 2) continue;
            const float* x = side ? v : k;
            if (j == 0) {
                ink_kv_sim(x, shadow_rows_, n, kvw, side ? kV4Block : dim_, 1, st_);
                x = shadow_rows_;
            }
            int8_t* codes = side ? sh.vc : sh.kc;
            uint16_t* sc = side ? sh.vs : sh.ks;
            if (!B.swa) {
                store_q8(x, codes, sc, kvw, pos_, n, st_);
            } else {
                const int a = (int) (pos_ % ring_), head = std::min(n, ring_ - a);
                store_q8(x, codes, sc, kvw, a, head, st_);
                if (head < n) store_q8(x + (size_t) head * kvw, codes, sc, kvw, 0, n - head, st_);
            }
        }
        ink_attend(q_, qgr, v4() == 2 ? sh.kc : B.k_codes, sh.vc, rel, qgr, fp(B.rel_table), n_rel_,
                   B.swa ? window_ : extent_, sink_, shadow_out_ + (size_t) j * rows_ * hq, part_, dim_, n_head_, n_kv_,
                   1.f / std::sqrt((float) dim_), pos_, n, B.swa ? window_ : 0, ring, st_, nullptr, 0,
                   v4() == 2 ? sh.ks : B.k_sc, sh.vs, 0, attn_counters_, ds_.on);
    }
}

// INK_KV4_CHECK: the real cache's attention output (att_) against the two shadows', summed by kind of call (a prompt
// batch or a token) and run (full or window); printed after each prompt batch and every 50 tokens.
void Inkling2::kv4_compare(const Block& B, int il, int n) {
    const size_t m = (size_t) n * n_head_ * dim_, hq = (size_t) n_head_ * dim_;
    std::vector<float> real(m), sim(m), raw(m);
    ck(cudaMemcpyAsync(real.data(), att_, m * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "INK_KV4_CHECK");
    ck(cudaMemcpyAsync(sim.data(), shadow_out_, m * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "INK_KV4_CHECK");
    ck(cudaMemcpyAsync(raw.data(), shadow_out_ + (size_t) rows_ * hq, m * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "INK_KV4_CHECK");
    ck(cudaStreamSynchronize(cs(st_)), "INK_KV4_CHECK");
    double* e = shadow_err_[n > 8 ? 0 : 1][B.swa ? 1 : 0];
    for (size_t i = 0; i < m; ++i) {
        const double a = real[i], b = sim[i], c = raw[i];
        e[0] += (a - b) * (a - b);
        e[1] += b * b;
        e[2] += (b - c) * (b - c);
        e[3] += (a - c) * (a - c);
        e[4] += c * c;
        e[5] = std::max(e[5], std::fabs(a - b));
    }
    if (il != n_layer_ - 1) return;
    static int tokens = 0;
    if (n <= 8 && ++tokens % 50 != 0) return;
    const char* kind[2] = {"prompt", "decode"};
    const char* run[2] = {"full", "window"};
    for (int c = n > 8 ? 0 : 1, r = 0; r < 2; ++r) {
        const double* x = shadow_err_[c][r];
        if (x[4] <= 0) continue;
        std::fprintf(stderr, "INK_KV4_CHECK %s to pos %lld, %s runs: real vs sim rel err %.3g (max abs %.3g); sim vs q8 "
                     "%.3g; real vs q8 %.3g\n", kind[c], (long long) (pos_ + n), run[r], std::sqrt(x[0] / x[1]), x[5],
                     std::sqrt(x[2] / x[4]), std::sqrt(x[3] / x[4]));
    }
}

// The experts' sum through its convolution, into the residual.
void Inkling2::layer_tail(Layer&, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const size_t keep = (size_t) (taps_ - 1), kvw = (size_t) n_kv_ * dim_;
    float* cur = r_ + (size_t) cur_ * n_embd_;
    const int rs = n_streams_ * n_embd_;
    if (stepping_ > 0) {   // step(): each row its sequence's state, from the offsets its head left (row_off_)
        ink_conv_rows(fo_, n_embd_, B.state + keep * (2 * kvw + n_embd_), row_off_, fp(B.ffn_conv), nullptr, 1, tmp_,
                      n_embd_, taps_, n, 1.f / std::sqrt((float) (taps_ + 1)), st_);
        ink_rows(tmp_, n_embd_, cur, rs, n_embd_, n, true, st_);
    } else {   // its output added to the residual as it is written
        ink_conv(fo_, B.state + keep * (2 * kvw + n_embd_), fp(B.ffn_conv), nullptr, 1, tmp_, n_embd_, taps_, n,
                 1.f / std::sqrt((float) (taps_ + 1)), st_,
                 snap_bits_ ? snap_at_ + (B.state + keep * (2 * kvw + n_embd_) - state_) : nullptr, rec_n_, snap_bits_,
                 0, cur, rs);
    }
    dbg("ffn", il, tmp_, n_embd_);
}

// The output reads the blocks and the stream through its own query over the depth, then its norm.
void Inkling2::output_rows(int first, int rows, float* h) {
    const int rs = n_streams_ * n_embd_;
    ink_mix(r_ + (size_t) first * rs, rs, out_mix_, fp(out_query_), 1, tmp_, (size_t) rows * n_embd_, n_embd_, rows,
            st_);
    rms_norm(tmp_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// A few tokens' logits: every token from the Q4_0 copy, then the shortlist's again from the Q8_0 matrix.  A prompt
// batch's rows (more than a window) take the Q8_0 matrix as they are.
void Inkling2::output_logits(const float* h, float* logits, int rows) {
    if (!head_short_) return lin(output_, h, logits, rows);
    lin(out_q4_, h, logits, rows);
    if (rows > std::max(opt_.window, 8)) return;   // more rows than the shortlist holds: the Q4_0 scores (step() only)
    ink_shortlist(logits, n_unpadded_, n_vocab_, kHeadMargin, head_ids_, head_count_, kHeadCap, rows, st_,
                  head_scratch_);
    ink_rescore((int) output_.type, head_dev_, h, n_embd_, head_ids_, head_count_, kHeadCap, logits, n_vocab_, rows,
                st_);
    if (head_exact_ && !recording_) head_check(logits, h, rows);
}

// INK_HEAD_CHECK=1: the exact logits beside the shortlist's, after the cap, as the softmax at temperature 1 sees them:
// the mean KL divergence and how often the top token is the same, printed every 64 tokens.
void Inkling2::head_check(const float* got, const float* h, int rows) {
    static double kl_sum = 0;
    static long n = 0, same = 0, over = 0;
    Dev exact = out_q4_;   // the Q8_0 matrix, read from host memory: slow, for the check only
    exact.p = const_cast<void*>(head_dev_);
    exact.type = output_.type;
    lin(exact, h, head_exact_, rows);
    std::vector<float> a((size_t) rows * n_vocab_), b(a.size());
    std::vector<int> cnt((size_t) rows);
    ck(cudaMemcpyAsync(a.data(), head_exact_, a.size() * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "check");
    ck(cudaMemcpyAsync(b.data(), got, b.size() * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)), "check");
    ck(cudaMemcpyAsync(cnt.data(), head_count_, rows * sizeof(int), cudaMemcpyDeviceToHost, cs(st_)), "check");
    ck(cudaStreamSynchronize(cs(st_)), "check");
    for (int r = 0; r < rows; ++r) {
        const float* x = a.data() + (size_t) r * n_vocab_;
        const float* y = b.data() + (size_t) r * n_vocab_;
        auto cap = [&](float v) { return softcap_ > 0 ? softcap_ * std::tanh(v / softcap_) : v; };
        double mx = -1e30, my = -1e30;
        int ax = 0, ay = 0;
        for (int i = 0; i < n_unpadded_; ++i) {
            if (cap(x[i]) > mx) mx = cap(x[i]), ax = i;
            if (cap(y[i]) > my) my = cap(y[i]), ay = i;
        }
        double zx = 0, zy = 0;
        for (int i = 0; i < n_unpadded_; ++i) zx += std::exp(cap(x[i]) - mx), zy += std::exp(cap(y[i]) - my);
        double kl = 0;
        for (int i = 0; i < n_unpadded_; ++i) {
            const double px = std::exp(cap(x[i]) - mx) / zx, lpx = cap(x[i]) - mx - std::log(zx);
            const double lpy = cap(y[i]) - my - std::log(zy);
            if (px > 0) kl += px * (lpx - lpy);
        }
        kl_sum += kl;
        same += ax == ay;
        over += cnt[(size_t) r] > kHeadCap;
        ++n;
    }
    if (n % 64 == 0)
        std::fprintf(stderr, "head check: %ld tokens, mean KL %.2e, same top token %.2f%%, shortlist over cap %ld\n", n,
                     kl_sum / n, 100.0 * same / n, over);
}

void Inkling2::finish_logits(float* logits, int rows) {
    if (softcap_ > 0.f) softcap(logits, (size_t) rows * n_vocab_, softcap_, st_);
    ink_fill_tail(logits, n_unpadded_, n_vocab_, rows, st_);   // the token table's padding is no token
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights from
// the bare gates, made to sum to one and scaled.
// --cache-bias: the chosen experts' weights as route() gives them, from the unbiased logits
void Inkling2::reweight(int, const float* logits, const int* ids, float* w) const {
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) sum += w[k] = 1.f / (1.f + std::exp(-logits[ids[k]]));
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

// INK_PREDICT_LOG: run il's choice for x from its file layer's router in the mapped file (F32), as route() picks
int Inkling2::predict_route(int il, const float* x, int* ranked, int n) const {
    if (il < 0 || il >= (int) blocks_.size()) return 0;
    const Block& B = blocks_[(size_t) il];
    const TensorInfo* t = gg_->find(blk(B.file, "ffn_gate_inp.weight"));
    if (!t || t->type != kTypeF32 || B.probs_b.size() != (size_t) n_expert_) return 0;
    const float* W = (const float*) home(*t).data(*t);   // n_expert rows of n_embd
    std::vector<std::pair<float, int>> pick((size_t) n_expert_);
    for (int e = 0; e < n_expert_; ++e) {
        const float* row = W + (size_t) e * n_embd_;
        float s = 0.f;
        for (int i = 0; i < n_embd_; ++i) s += row[i] * x[i];
        pick[(size_t) e] = {1.f / (1.f + std::exp(-s)) + B.probs_b[(size_t) e], e};
    }
    n = std::min(n, n_expert_);
    std::partial_sort(pick.begin(), pick.begin() + n, pick.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int k = 0; k < n; ++k) ranked[k] = pick[(size_t) k].second;
    return n;
}

void Inkling2::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = std::max(sum, kMinWeightSum);
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
    // INK_ROUTE_LOG=FILE: a line a token and run - position, run, the file's layer, then expert:weight:in VRAM
    static FILE* log = [] {
        const char* f = std::getenv("INK_ROUTE_LOG");
        return f ? std::fopen(f, "w") : nullptr;
    }();
    if (log) {
        static std::mutex m;
        const std::lock_guard<std::mutex> lk(m);
        const Layer& X = layers_[(size_t) ex(il)];
        std::fprintf(log, "%lld %d %d", (long long) pos_, il, blocks_[(size_t) il].file);
        for (int k = 0; k < n_used_; ++k)
            std::fprintf(log, " %d:%.3f:%d", ids[k], w[k], X.slot.empty() ? 0 : X.slot[(size_t) ids[k]] != nullptr);
        std::fputc('\n', log);
    }
}

bool Inkling2::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, B.probs_b_dev.p ? fp(B.probs_b_dev) : nullptr, nullptr, true,
               w_scale_, ids, w, st_);
    return true;
}

void Inkling2::lin_f32(const Dev& w, const float* x, float* y, int n) {
    if (n >= kStageRows && w.type == kTypeF32) return gemm_f32_blas(blas_, fp(w), x, y, w.n_in, w.n_out, n);
    if (w.type == kTypeF32 && w.scale == 1.0f && ink_gemv_f32(fp(w), x, y, w.n_in, w.n_out, n, st_)) return;
    lin(w, x, y, n);
}

// The router's F32 matrix on the experts' input; a prompt batch's through SGEMM.
void Inkling2::router_scores(const Layer& L, int n) { lin_f32(L.gate_inp, xn_, rl_, n); }

void Inkling2::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

// The recorded layers' launches with programmatic dependent launch (pdl.hpp) on sm_90+: every kernel of Inkling's
// that is launched with launch_pdl waits (pdl_wait) before it reads or writes anything; the attention's kernels are
// launched as before.  INK_PDL=0 turns it off.
bool Inkling2::pdl_graphs() const {
    static const bool on = [] {
        const char* e = std::getenv("INK_PDL");
        if (e && std::atoi(e) == 0) return false;
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        return major >= 9;
    }();
    return on;
}

// A layer's tail (the experts' sum's convolution into the residual) and the next layer's head recorded as one: the
// card goes on to the next layer as soon as the experts' sum is in.  INK_MERGE=0 turns it off.
bool Inkling2::merge_heads() const {
    static const bool on = [] {
        const char* e = std::getenv("INK_MERGE");
        return !(e && std::atoi(e) == 0);
    }();
    return on;
}

int Inkling2::head_key(int il, int n) const { return blocks_[(size_t) il].swa ? 0 : room_key(n); }

// ---- a sequence's state: the convolutions' past inputs (state_), the window runs' rings (extra), the full runs'
// rows

std::vector<std::pair<uint8_t*, size_t>> Inkling2::arrays(const Block& B) const {
    const size_t row = (size_t) n_kv_ * dim_;
    if (q8())
        return {{(uint8_t*) B.k_codes, v4() == 2 ? row / 2 : row}, {(uint8_t*) B.k_sc, row / 32 * sizeof(uint16_t)},
                {(uint8_t*) B.v_codes, v4() ? row / 2 : row}, {(uint8_t*) B.v_sc, row / 32 * sizeof(uint16_t)}};
    return {{(uint8_t*) B.keys, row * sizeof(uint16_t)}, {(uint8_t*) B.vals, row * sizeof(uint16_t)}};
}

size_t Inkling2::extra_bytes() const {
    size_t b = 0;
    for (const Block& B : blocks_)
        if (B.swa)
            for (const auto& [p, row] : arrays(B)) b += (size_t) keep_ * row;
    return b;
}

void Inkling2::save_layer(int il, int64_t pos, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    ck(cudaMemcpy(dst + (size_t) (B.state - state_) * sizeof(float), B.state, state_per_ * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "save state");
    if (!B.swa) return;
    uint8_t* d = dst + state_n_ * sizeof(float);
    for (int j = 0; j < il; ++j)
        if (blocks_[(size_t) j].swa)
            for (const auto& [p, row] : arrays(blocks_[(size_t) j])) d += (size_t) keep_ * row;
    for (const auto& [cache, row] : arrays(B)) {
        std::memset(d, 0, (size_t) keep_ * row);   // the places of positions before the sequence's start
        for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
            const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
            ck(cudaMemcpy(d + (size_t) (p - (pos - keep_)) * row, cache + (size_t) at * row, (size_t) m * row,
                          cudaMemcpyDeviceToHost),
               "save state");
            p += m;
        }
        d += (size_t) keep_ * row;
    }
}

void Inkling2::load_extra(const uint8_t* src, int64_t pos) {
    // Target snapshots omit DSpark's features and caches. Rebuild their context from the next row.
    if (ds_.on) ds_.lo = ds_.hi = pos;
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        for (const auto& [cache, row] : arrays(B)) {
            for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
                const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
                ck(cudaMemcpy(cache + (size_t) at * row, src + (size_t) (p - (pos - keep_)) * row, (size_t) m * row,
                              cudaMemcpyHostToDevice),
                   "load state");
                p += m;
            }
            src += (size_t) keep_ * row;
        }
    }
}

size_t Inkling2::card_row_bytes() const {
    size_t b = 0;
    for (const Block& B : blocks_)
        if (!B.swa)
            for (const auto& [p, row] : arrays(B)) b += row;
    return b;
}

void Inkling2::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    for (const Block& B : blocks_) {
        if (B.swa || n == 0) continue;
        for (const auto& [cache, row] : arrays(B)) {
            ck(cudaMemcpy(dst, cache + (size_t) first * row, (size_t) n * row, cudaMemcpyDeviceToHost), "save rows");
            dst += (size_t) n * row;
        }
    }
}

void Inkling2::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    for (const Block& B : blocks_) {
        if (B.swa || n == 0) continue;
        for (const auto& [cache, row] : arrays(B)) {
            ck(cudaMemcpy(cache + (size_t) first * row, src + (size_t) (first - from) * row, (size_t) n * row,
                          cudaMemcpyHostToDevice),
               "load rows");
            src += (size_t) count * row;
        }
    }
}

}  // namespace strata::poly
