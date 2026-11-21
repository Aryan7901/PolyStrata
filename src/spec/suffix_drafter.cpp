// src/spec/suffix_drafter.cpp - see include/strata/spec/suffix_drafter.hpp.
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::spec {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}
}  // namespace

SuffixDrafter::SuffixDrafter(int min_match, int max_match, size_t capacity_tokens)
    : min_match_(std::max(3, min_match)), max_match_(std::max(min_match, max_match)) {
    size_t cap = 1;
    while (cap < capacity_tokens * 2) cap <<= 1;          // load factor <= 0.5 at the nominal capacity
    table_.assign(cap, Slot{});
    mask_ = cap - 1;
    if (const char* v = std::getenv("STRATA_SUFFIX_GRAM8"); v && std::atoi(v) == 0) gram8_ = false;
    if (gram8_) table8_.assign(cap, Slot{});
    hist_.reserve(capacity_tokens);
}

void SuffixDrafter::reset() {
    hist_.clear();
    std::fill(table_.begin(), table_.end(), Slot{});
    std::fill(table8_.begin(), table8_.end(), Slot{});
    last_match_ = 0;
}

uint64_t SuffixDrafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;   // never 0
}

uint64_t SuffixDrafter::key8_at(size_t end) const {
    uint64_t h = 0x9E3779B97F4A7C15ull;
    for (int i = 0; i < GRAM; ++i) h = mix(h ^ (uint32_t) hist_[end - (size_t) i]) + 0x632BE59BD9B4E019ull;
    return h | 1ull;
}

SuffixDrafter::Slot* SuffixDrafter::find_slot(uint64_t key, bool insert) { return find_slot_in(table_, mask_, key, insert); }

SuffixDrafter::Slot* SuffixDrafter::find_slot_in(std::vector<Slot>& t, size_t mask, uint64_t key, bool insert) {
    for (size_t i = key & mask, probes = 0; probes <= mask; i = (i + 1) & mask, ++probes) {
        Slot& s = t[i];
        if (s.key == key) return &s;
        if (s.key == 0) {
            if (!insert) return nullptr;
            s.key = key;
            return &s;
        }
    }
    return nullptr;                                         // table full: the history outgrew its capacity
}

void SuffixDrafter::append(const int32_t* tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        hist_.push_back(tokens[i]);
        const size_t end = hist_.size() - 1;
        if (gram8_ && end >= (size_t) GRAM - 1) {
            if (Slot* g = find_slot_in(table8_, mask_, key8_at(end), true)) {
                for (int w = WAYS - 1; w > 0; --w) g->pos[w] = g->pos[w - 1];
                g->pos[0] = (uint32_t) end;
                if (g->n < WAYS) ++g->n;
            }
        }
        if (end < 2) continue;
        Slot* s = find_slot(key_at(end), true);
        if (s == nullptr) continue;
        for (int w = WAYS - 1; w > 0; --w) s->pos[w] = s->pos[w - 1];
        s->pos[0] = (uint32_t) end;
        if (s->n < WAYS) ++s->n;
    }
}

int SuffixDrafter::propose(int max_k, int32_t* out) {
    last_match_ = 0;
    const size_t n = hist_.size();
    if (n < 4 || max_k <= 0) return 0;
    const size_t cur = n - 1;
    const Slot* s = find_slot(key_at(cur), false);
    const Slot* g = gram8_ && cur >= (size_t) GRAM - 1 ? find_slot_in(table8_, mask_, key8_at(cur), false) : nullptr;
    if (s == nullptr && g == nullptr) return 0;
    size_t best_end = 0;
    int best_len = 0;
    // the trigram's most recent ends, then the 8-gram's: a trigram seen often keeps only its newest, the 8-gram's
    // older ends are the long repeats
    for (int pass = 0; pass < 2; ++pass) {
        const Slot* c = pass == 0 ? s : g;
        if (c == nullptr) continue;
        for (int w = 0; w < c->n; ++w) {
            const size_t p = c->pos[w];
            if (p >= cur) continue;                         // the current suffix itself
            int len = 0;
            while (len < max_match_ && len <= (int) p && hist_[p - len] == hist_[cur - len]) ++len;
            if (len > best_len || (len == best_len && p > best_end)) { best_len = len; best_end = p; }
        }
    }
    if (best_len < min_match_) return 0;
    last_match_ = best_len;
    int k = 0;
    // The continuation may run into the current suffix (periodic text); reading history up to `cur` is valid.
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = hist_[q];
    return k;
}

int SuffixDrafter::propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out) {
    if (n_extra <= 0) return propose(max_k, out);
    // the index holds the history's trigrams only: appended without indexing, the extra tokens just form the suffix
    const size_t n0 = hist_.size();
    hist_.insert(hist_.end(), extra, extra + n_extra);
    const int k = propose(max_k, out);
    hist_.resize(n0);
    return k;
}

}  // namespace strata::spec
