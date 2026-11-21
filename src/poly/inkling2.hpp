// src/poly/inkling2.hpp - Thinking Machines' Inkling 2 Mini (llama.cpp architecture "inkling2"): 24 layers of
// weights, of which the middle ones run three times (the loop), with routed experts and no shared one.
//
// The engine sees a layer a run: block_count is the runs (48), and file_layer() names the file's layer whose
// weights a run uses.  A run that repeats an earlier one's layer (a "twin") shares that layer's weights on the card;
// its attention caches and convolution state are its own, since every run computes its own keys and values.  A
// twin's experts are the same bytes of the file as its first run's.
//
// What a layer does, after llama.cpp's graph (src/models/inkling2.cpp):
//  - No rotary positions: a learned bias by distance (a table of n_rel terms a distance, read through n_rel terms a
//    head the token projects) carries them, and each token gives each head a sink logit of its own.  Window layers
//    attend the last 1024 positions; the full ones every position, the bias only up to the table's extent.
//  - Keys, values, the attention's output and the experts' sum each go through a causal short convolution with a
//    residual; the full layers' key and value convolutions also take taps the token projects.
//  - Attention over the depth: every 6 layers the residual starts again from the attention's output, the stream
//    before it kept as a block; a layer's attention and experts read the blocks and the stream through learned
//    queries over their RMS norms.  The streams the model keeps are r_'s copies (streams()), so the engine's
//    layer-at-a-time prompt reads keep them across layers as they keep the residual.
//  - The loop: each later pass restarts from the state before the loop and reads the pass before's blocks through
//    gated messages into its attention's three inputs and its experts' input.
#pragma once

#include "engine.hpp"

#include <array>
#include <initializer_list>

namespace strata::poly {

class Igpu;

class Inkling2 final : public Engine {
public:
    ~Inkling2() override;
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override;
    bool snaps_spread() const override { return true; }   // the convolutions leave their state after any token
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return n_streams_; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    int predict_route(int il, const float* x, int* ranked, int n) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void router_scores(const Layer& L, int n) override;
    void finish_logits(float* logits, int rows) override;
    void output_logits(const float* h, float* logits, int rows) override;
    bool output_in_vram() const override { return !head_short_; }
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    int head_key(int il, int n) const override;
    int file_layer(int il) const override;
    bool batch_heads() const override { return true; }
    bool sees() const override { return true; }   // pictures: rows of Inkling's vision encoder (strata-vision)
    void after_load() override;

private:
    /// A run's weights beside the ones every architecture has (Layer), and its caches and state.
    struct Block {
        bool swa = false;
        int file = 0;          ///< the file's layer
        int pass = 0;          ///< the loop's pass, 0 outside it
        bool twin = false;     ///< repeats an earlier run's layer: its weights are that run's
        Dev wq, wk, wv, wo, wgate, wrel, wsink, kconv_proj, vconv_proj;
        Dev q_clip, rel_clip, rel_table, k_conv, v_conv, attn_conv, ffn_conv;
        Dev q_norm, k_norm, v_norm;   ///< the full layers' three input norms (a window layer's is L.attn_norm)
        Dev attn_query, ffn_query;    ///< attention over the depth: 1 or 3 queries, and 1
        Dev loop_down[4], loop_gain[4], loop_forget[4];   ///< the loop's message controllers: q, k, v, experts
        Dev wqgr;          ///< attn_q, attn_gate and attn_rel stacked: one product of the queries' input
        Dev wall;          ///< a window layer's wqgr, attn_k and attn_v stacked (wqgr, wk and wv are its slices): one
                           ///< product where all five read one input (a run without the loop's messages)
        Dev loop_gf[4];    ///< each controller's gain and forget stacked: one product of its h
        std::vector<float> probs_b;
        Dev probs_b_dev;
        uint16_t* keys = nullptr;   ///< fp16 rows of n_kv * dim: a ring in a window layer
        uint16_t* vals = nullptr;
        int8_t* k_codes = nullptr;  ///< --kv q8_0: the same rows as int8 codes and an fp16 scale every 32 values;
        int8_t* v_codes = nullptr;  ///< k8v4: the values' as ink_store_q4's 4-bit codes (half the bytes), k4v4: both
        uint16_t* k_sc = nullptr;
        uint16_t* v_sc = nullptr;
        float* state = nullptr;     ///< the convolutions' past inputs: k, v, the attention's output, the experts' sum
        // what the run does with the depth's streams (slots of r_)
        InkStates attn_mix{}, ffn_mix{};
        int push = -1;          ///< the slot the stream before the attention goes to, starting a block; -1: none
        bool restart = false;   ///< a later pass's first run: the bank is taken and the stream restarts
        int msg_row = 0;        ///< a later pass: the first of its two rows of the message queries
    };

    void attention(Layer& L, Block& B, int n);
    /// Matrices of one input stacked as one (their rows one after the other), uploaded as they are: one product
    /// for all of them.  The same type each, n_in values a row.
    Dev fused(int il, std::initializer_list<const char*> names, int64_t n_in);
    Dev msg_queries();
    /// The loop's message controller of module m on rows of x (n_embd a token, contiguous), in place.
    void control(Block& B, int m, float* x, const float* msg, int n);
    /// The attention's three controllers (modules 0 to 2) together: out[j] from x[j] and msg[j].
    void control3(Block& B, float* const* x, const float* const* msg, float* const* out, int n);
    void plan();
    /// The cache is codes and their scales (q8_0, k8v4, k4v4), not fp16 rows.
    bool q8() const { return opt_.kv_format == KvFmt::kQ8_0 || v4(); }
    /// Its 4-bit sides, as ink_attend takes them: 0 none, 1 the values (k8v4), 2 the keys too (k4v4).
    int v4() const { return opt_.kv_format == KvFmt::kK8V4 ? 1 : opt_.kv_format == KvFmt::kK4V4 ? 2 : 0; }
    /// A run's cache as arrays of rows, keys' side first: (the array, its bytes a row).  fp16: one a side; q8_0, k8v4
    /// and k4v4: the codes and the scales a side.
    std::vector<std::pair<uint8_t*, size_t>> arrays(const Block& B) const;
    /// INK_KV4_CHECK (inkling2.cpp): Q8_0 caches beside a 4-bit one, of the simulated ([0]) and of the plain ([1])
    /// rows of its 4-bit sides, and the attention over them.
    struct Shadow {
        int8_t* kc = nullptr;
        int8_t* vc = nullptr;
        uint16_t* ks = nullptr;
        uint16_t* vs = nullptr;
    };
    std::vector<std::array<Shadow, 2>> shadow_;
    float* shadow_rows_ = nullptr;
    float* shadow_out_ = nullptr;
    double shadow_err_[2][2][6] = {};
    void kv4_shadow(const Block& B, int il, int n, const float* k, const float* v, int qgr, const float* rel, int ring);
    void kv4_compare(const Block& B, int il, int n);
    /// lin() of an F32 matrix, a prompt batch through cuBLAS's SGEMM.
    void lin_f32(const Dev& w, const float* x, float* y, int n);

    std::vector<Block> blocks_;
    std::vector<uint8_t> swa_;
    int n_head_ = 0, n_kv_ = 0, dim_ = 0, n_rel_ = 0, extent_ = 0, window_ = 0, taps_ = 0, block_size_ = 0;
    int core_start_ = 0, core_end_ = 0, passes_ = 1, gate_rank_ = 0, n_file_ = 0, n_unpadded_ = 0;
    float qk_clip_ = 0, rel_clip_ = 0, log_floor_ = 0, log_alpha_ = 0, softcap_ = 0, w_scale_ = 0;
    int ring_ = 0, keep_ = 0;
    size_t state_per_ = 0;    ///< floats of a run's convolution state
    // the depth's streams: r_'s copies of n_embd values a token
    int n_streams_ = 0;
    int cur_ = 0;                       ///< the stream (always slot 0)
    std::vector<int> block_slot_;       ///< by the file layer that starts a block: its slot, or -1
    std::vector<std::pair<int, int>> bank_;   ///< a later pass's first run copies these slots (from, to)
    int bank_last_ = -1, prelude_ = -1; ///< the slot of the stream at the end of the pass before; of the loop's start
    InkStates bank_states_{};           ///< what the messages read
    InkStates out_mix_{};               ///< the output's attention over the depth
    Dev tok_norm_, out_query_, msg_query_, ones_;
    // the output's shortlist (INK_HEAD=0 turns it off): a Q4_0 copy of the output matrix scores every token, and the
    // tokens within kHeadMargin of the best are scored again from the Q8_0 matrix
    Dev out_q4_;
    bool head_short_ = false;
    int* head_ids_ = nullptr;
    int* head_count_ = nullptr;
    float* head_exact_ = nullptr;   // INK_HEAD_CHECK=1: the exact logits beside, to compare
    float* head_scratch_ = nullptr;
    std::unique_ptr<Igpu> igpu_;    // INK_IGPU=1: the integrated GPU that computes some of the CPU's experts
    void* head_host_ = nullptr;     // the Q8_0 output matrix in pinned host memory, mapped: the shortlist reads it
    const void* head_dev_ = nullptr;   // the card's address of it
    void head_check(const float* got, const float* h, int rows);

    // device scratch, rows_ tokens
    float *mix_ = nullptr, *msg_ = nullptr, *ctl_ = nullptr, *ctl_h_ = nullptr,
          *ctl_g_ = nullptr, *ctl_f_ = nullptr, *xin_ = nullptr, *q_ = nullptr, *gate_ = nullptr, *rel_ = nullptr,
          *sink_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *vraw_ = nullptr, *v_ = nullptr, *dyn_ = nullptr,
          *att_ = nullptr, *part_ = nullptr, *tmp_ = nullptr;
    float* xq3_ = nullptr;           ///< the attention's inputs in Q8_1 (layer_head)
    long long* row_off_ = nullptr;   ///< step(): each row's sequence's state, in floats from the bound one's
};

}  // namespace strata::poly
