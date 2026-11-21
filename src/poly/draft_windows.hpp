// src/poly/draft_windows.hpp - a round's window when the drafter states each draft's chance (DSpark): how many of its
// drafts to verify, or the lookup window in their place.
//
// Adapted from Basalt's DraftPolicy (include/basalt/spec/draft_policy.hpp and src/spec/draft_policy.cpp,
// https://github.com/jesdga95/basalt, MIT, Copyright the Basalt authors), which grew out of the same Strata policy
// as strata/spec/draft_policy.hpp.  What it does:
//
//   drafter E(t) = 1 + q(c_0) + q(c_0) q(c_1) + ... (t - 1 terms), q = the acceptance rate of drafts whose stated
//           chance was about c (decayed counts per chance bin, the stated chance as the prior), at the measured cost
//           of a window of t
//   lookup  E(k) = 1 + q + q^2 + ... + q^k, q = the acceptance of lookup drafts whose match was about as long (4
//           buckets of match length), at the measured cost of k + 1
//
// and the window is the one with the most expected tokens per millisecond; the lookup window only when it beats the
// drafter's best by `margin`.  A round's cost is scale x base(size): the scale follows every round fast (the context
// grows, the expert cache moves), each size's base follows its own rounds slowly, and sizes not measured lately are
// inferred from the ones that were.  So a row costs what it costs on this machine: with experts on the CPU every
// extra verified row brings experts of its own, and a window grows only where its drafts pay for them.  It only
// chooses which drafts are verified: the output is the same.
//
// Nothing here knows the model or the engine.
#pragma once

#include <algorithm>
#include <array>

namespace strata::poly {

class DraftWindows {
public:
    static constexpr int kMaxT = 16;
    static constexpr int kBuckets = 4;
    static constexpr int kChanceBins = 9;

    explicit DraftWindows(int max_t, double margin = 0.03) : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {}

    struct Pick {
        bool lookup = false;
        int t = 1;   // window size (1 + drafts)
    };

    /// The window (1 + up to `n` drafts) with the most expected tokens per ms.
    int window(const float* chance, int n) const {
        const int most = std::clamp(n + 1, 1, max_t_);
        double e = 1.0, all = 1.0, best = 1.0 / cost_ms(1);
        int best_t = 1;
        for (int t = 2; t <= most; ++t) {
            all *= chance_rate(chance[t - 2]);
            e += all;
            const double r = e / cost_ms(t);
            if (r > best) {
                best = r;
                best_t = t;
            }
        }
        // a size whose cost is only guessed (or old) is tried a few times when its drafts are likely accepted, so a
        // guessed or stale cost cannot keep the window from ever growing
        if (best_t < most && unmeasured(best_t + 1)) {
            double joint = 1.0;
            for (int i = 0; i < best_t; ++i) joint *= chance_rate(chance[i]);
            if (joint >= kProbeChance) ++best_t;
        }
        return best_t;
    }

    /// Whether window(chance, n) can hold draft j (from 0) when only the chances of drafts 0..j are known: the later
    /// ones are taken at the best rate a stated chance can have.  False means every window of those drafts leaves
    /// draft j out, so a drafter that makes its drafts in turn can stop before the next one.
    bool may_verify(const float* chance, int j, int n) const {
        const int most = std::clamp(n + 1, 1, max_t_);
        if (j < 0 || j + 2 > most) return false;
        double e = 1.0, all = 1.0, best = 1.0 / cost_ms(1);
        int best_t = 1;
        for (int t = 2; t <= j + 1; ++t) {
            all *= chance_rate(chance[t - 2]);
            e += all;
            const double r = e / cost_ms(t);
            if (r > best) {
                best = r;
                best_t = t;
            }
        }
        const double hi = best_rate();
        const double joint = all * chance_rate(chance[j]);
        for (int t = j + 2; t <= most; ++t) {
            all *= t == j + 2 ? chance_rate(chance[j]) : hi;
            e += all;
            if (e / cost_ms(t) > best) return true;
        }
        return best_t == j + 1 && unmeasured(j + 2) && joint >= kProbeChance;
    }

    /// The drafter's window against a lookup proposal of `lookup_k` tokens behind a match of `match`.
    Pick choose_window(const float* chance, int n, int lookup_k, int match) const {
        Pick p;
        p.t = window(chance, n);
        if (lookup_k <= 0) return p;
        return lookup_over(p.t, drafter_tokens(chance, p.t) / cost_ms(p.t), lookup_k, match);
    }

    /// Without drafts: the lookup window or a token alone.
    Pick choose_lookup(int lookup_k, int match) const {
        if (lookup_k <= 0) return Pick{};
        return lookup_over(1, 1.0 / cost_ms(1), lookup_k, match);
    }

    /// After a drafter window: the chances of its t - 1 drafts, the drafts accepted, and the round's time.
    void observe_window(const float* chance, int t, int accepted, double round_ms) {
        // drafts 0..accepted-1 were accepted given their prefix and draft `accepted` (if the window had it) was not;
        // the drafts after it were never tested
        const int tested = std::min(t - 1, accepted + 1);
        for (int i = 0; i < tested; ++i) {
            const int b = chance_bin(chance[i]);
            c_ok_[b] = kChanceDecay * c_ok_[b] + (i < accepted ? 1.0 : 0.0);
            c_n_[b] = kChanceDecay * c_n_[b] + 1.0;
        }
        observe(false, t, accepted, 0, round_ms);
    }

    /// After any round: the window it used, the drafts accepted, the lookup match (lookup rounds) and its time
    /// (0: not timed, as a round that recorded launches).
    void observe(bool lookup, int t, int accepted, int match, double round_ms) {
        t = std::clamp(t, 1, kMaxT);
        ++round_;
        if (round_ms > 0) {
            if (scale_ <= 0) scale_ = round_ms / kShape[t];   // the first round sets the scale
            if (!recent(t)) {                                 // new, or measured too long ago: from what the others imply
                base_[t] = base(t);
                base_n_[t] = kBasePrior;
            }
            scale_ += kScaleAlpha * (round_ms / base_[t] - scale_);
            const double w = std::max(kCostAlpha, 1.0 / (base_n_[t] + 1.0));
            base_[t] = (1.0 - w) * base_[t] + w * round_ms / scale_;
            base_n_[t] += 1.0;
            seen_[t] = round_;
        }
        if (lookup) {
            const int b = bucket(match);
            ok_[b] = kDecay * ok_[b] + accepted;
            bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
        }
    }

    double lookup_rate(int match) const {
        const int b = bucket(match);
        return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
    }
    double chance_rate(float chance) const {
        const int b = chance_bin(chance);
        const double c = std::clamp((double) chance, 0.0, 1.0);
        return (c_ok_[b] + kChancePriorN * c) / (c_n_[b] + kChancePriorN);
    }
    /// Measured or scaled round time of a window of t tokens.
    double cost_ms(int t) const {
        t = std::clamp(t, 1, kMaxT);
        return (scale_ > 0 ? scale_ : 1.0) * base(t);
    }
    /// The acceptance seen in chance bin b and the drafts behind it (decayed), for a report.
    double bin_rate(int b) const { return c_n_[(size_t) b] > 0 ? c_ok_[(size_t) b] / c_n_[(size_t) b] : 0.0; }
    double bin_n(int b) const { return c_n_[(size_t) b]; }

private:
    // The shape of a round's cost by window size, relative to one token, used only below the smallest measured size
    // or while a single size is measured.  Here a verified row costs about 0.55 of a token (its experts are mostly
    // new, many on the CPU), steeper than Basalt's prior.
    static constexpr double kShape[kMaxT + 1] = {0.0, 1.0, 1.55, 2.1, 2.65, 3.2, 3.75, 4.3, 4.85,
                                                 5.4, 5.95, 6.5, 7.05, 7.6, 8.15, 8.7, 9.25};
    static constexpr double kScaleAlpha = 0.2;
    static constexpr double kCostAlpha = 0.1;    // EMA weight of a size's new base
    static constexpr double kBasePrior = 3.0;    // a size's first base, worth this many rounds
    static constexpr long kStale = 64;           // rounds after which a size's base is from another time
    static constexpr double kDecay = 0.97;       // lookup counts fade
    static constexpr double kPriorQ[kBuckets] = {0.75, 0.88, 0.93, 0.96};
    static constexpr double kPriorN = 4.0;
    static constexpr int kProbes = 3;            // a size is tried this often before its guessed cost can veto it
    // stated chances: the bins' upper edges, finest near 1, where long windows are decided
    static constexpr float kChanceEdge[kChanceBins - 1] = {0.3f, 0.5f, 0.7f, 0.8f, 0.9f, 0.95f, 0.98f, 0.995f};
    static constexpr double kChanceDecay = 0.98;
    static constexpr double kChancePriorN = 10.0;
    static constexpr double kProbeChance = 0.85;   // the joint chance at which a window one row past the best is probed

    static int bucket(int match) { return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3; }
    static int chance_bin(float chance) {
        int b = 0;
        while (b < kChanceBins - 1 && chance >= kChanceEdge[b]) ++b;
        return b;
    }
    // each bin's rate at its top edge (a chance of 1 in the last bin): no stated chance gets more
    double best_rate() const {
        double hi = 0.0;
        for (int b = 0; b < kChanceBins; ++b) {
            const double top = b < kChanceBins - 1 ? (double) kChanceEdge[b] : 1.0;
            hi = std::max(hi, (c_ok_[b] + kChancePriorN * top) / (c_n_[b] + kChancePriorN));
        }
        return hi;
    }
    bool recent(int t) const { return base_n_[t] > 0 && round_ - seen_[t] <= kStale; }
    bool unmeasured(int t) const { return base_n_[t] < kBasePrior + kProbes || !recent(t); }
    double base(int t) const {
        if (recent(t)) return base_[t];
        for (const bool lately : {true, false}) {
            const double b = base_from(t, lately);
            if (b > 0) return b;
        }
        return kShape[t];
    }
    // the measured sizes next to t: between two, the line through them; above the largest, its line on from the two
    // largest (the cost of a row on this machine); below the smallest, or with one, the prior shape scales it
    double base_from(int t, bool lately) const {
        int lo = 0, hi = 0, below = 0;
        for (int u = 1; u <= kMaxT; ++u) {
            if (u == t || base_n_[u] <= 0 || (lately && !recent(u))) continue;
            if (u < t) {
                below = lo;
                lo = u;
            } else if (hi == 0) {
                hi = u;
            }
        }
        if (lo > 0 && hi > 0) return base_[lo] + (base_[hi] - base_[lo]) * (double) (t - lo) / (double) (hi - lo);
        if (lo > 0 && below > 0)
            return base_[lo] + std::max(0.0, (base_[lo] - base_[below]) / (double) (lo - below)) * (double) (t - lo);
        if (lo > 0) return base_[lo] * kShape[t] / kShape[lo];
        if (hi > 0) return base_[hi] * kShape[t] / kShape[hi];
        return -1.0;
    }
    double drafter_tokens(const float* chance, int t) const {
        double e = 1.0, all = 1.0;
        for (int i = 0; i + 1 < t; ++i) {
            all *= chance_rate(chance[i]);
            e += all;
        }
        return e;
    }
    // `rate`: the expected tokens per ms of the window of t the drafter (or a token alone) would run
    Pick lookup_over(int t, double rate, int lookup_k, int match) const {
        Pick p;
        p.t = t;
        const double q = lookup_rate(match);
        double e = 1.0, qi = 1.0, best = 0.0;
        int best_t = 0;
        for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
            qi *= q;
            e += qi;
            const double r = e / cost_ms(k + 1);
            if (r > best) {
                best = r;
                best_t = k + 1;
            }
        }
        if (best_t > 0 && best > rate * (1.0 + margin_)) {
            p.lookup = true;
            p.t = best_t;
            return p;
        }
        // a confident lookup that needs a size not measured yet (or not for long) tries it a few times
        const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
        if (t_full > p.t && unmeasured(t_full) && q >= 0.85) {
            p.lookup = true;
            p.t = t_full;
        }
        return p;
    }

    int max_t_;
    double margin_;
    double scale_ = 0.0;                                   // round ms = scale x base(size)
    std::array<double, kMaxT + 1> base_{}, base_n_{};
    std::array<long, kMaxT + 1> seen_{};
    long round_ = 0;
    std::array<double, kBuckets> ok_{}, bad_{};            // lookup drafts accepted / windows cut short, decayed
    std::array<double, kChanceBins> c_ok_{}, c_n_{};       // drafter drafts accepted / tested, by stated chance
};

}  // namespace strata::poly
