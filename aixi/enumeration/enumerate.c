// Exhaustive enumeration of a small Brainfuck-family machine (in the style of F-ICL's machine F).
// Instructions: < > + [ ]   on a bit tape of width W.
// Input: n=4 bits preloaded at cells IN0..IN0+3, head starts at IN0, all other cells 0.
// Output: the bit under the head at halt.
// A program is kept only if it halts within T steps and never leaves the tape, on ALL 16 inputs.
// Its behaviour is the 16-entry truth table (uint16). Prior weight w(p) = 5^{-|p|}.
// Per behaviour we aggregate: total weight, shortest length, runtimes, Levin Kt, a witness program.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define W 10
#define IN0 3
#define NIN 16
#define T 128
#define MAXL 16
#ifndef OUTMODE
#define OUTMODE 0   // 0: output = bit under head at halt; 1: output = bit in the first input cell at halt
#endif

static const char SYM[5] = {'<','>','+','[',']'};

typedef struct {
    double weight;          // sum of 5^-L over programs computing it
    int K;                  // shortest program length
    double count;           // number of programs
    double w_meanT;         // weight-weighted sum of mean runtime
    double w_maxT;          // weight-weighted sum of max runtime
    double Kt_mean;         // min over programs of L*log2(5) + log2(mean runtime)
    double Kt_max;          // min over programs of L*log2(5) + log2(max runtime)
    double short_min_meanT; // min mean runtime among shortest programs
    double short_min_maxT;  // min max runtime among shortest programs
    char witness[MAXL+1];
    double cnt_by_L[MAXL];
} Beh;

static Beh beh[65536];

int main(int argc, char **argv) {
    int Lmax = argc > 1 ? atoi(argv[1]) : 10;
    const char *out = argc > 2 ? argv[2] : "behaviours.csv";
    for (int b = 0; b < 65536; b++) { beh[b].K = 999; beh[b].Kt_mean = 1e9; beh[b].Kt_max = 1e9; beh[b].short_min_meanT = 1e9; beh[b].short_min_maxT = 1e9; }

    long long total_valid_bracket = 0, total_kept = 0;
    int prog[MAXL], match[MAXL], stack[MAXL];
    double log2_5 = log2(5.0);

    for (int L = 0; L <= Lmax; L++) {
        long long nprog = 1; for (int i = 0; i < L; i++) nprog *= 5;
        long long kept_L = 0, valid_L = 0;
        double wL = pow(5.0, -L);
        for (long long code = 0; code < nprog; code++) {
            long long c = code;
            for (int i = 0; i < L; i++) { prog[i] = (int)(c % 5); c /= 5; }
            // bracket matching
            int sp = 0, ok = 1;
            for (int i = 0; i < L && ok; i++) {
                if (prog[i] == 3) stack[sp++] = i;
                else if (prog[i] == 4) { if (sp == 0) ok = 0; else { int j = stack[--sp]; match[i] = j; match[j] = i; } }
            }
            if (!ok || sp != 0) continue;
            valid_L++;
            // run on all inputs
            unsigned tt = 0; int steps[NIN]; int good = 1;
            for (int x = 0; x < NIN && good; x++) {
                unsigned char tape[W]; memset(tape, 0, W);
                for (int k = 0; k < 4; k++) tape[IN0 + k] = (x >> k) & 1;
                int head = IN0, pc = 0, t = 0;
                while (pc < L) {
                    if (t >= T) { good = 0; break; }
                    t++;
                    int op = prog[pc];
                    if (op == 0) { head--; if (head < 0) { good = 0; break; } pc++; }
                    else if (op == 1) { head++; if (head >= W) { good = 0; break; } pc++; }
                    else if (op == 2) { tape[head] ^= 1; pc++; }
                    else if (op == 3) { if (tape[head] == 0) pc = match[pc] + 1; else pc++; }
                    else { if (tape[head] == 1) pc = match[pc] + 1; else pc++; }
                }
                if (!good) break;
                steps[x] = t;
                if ((OUTMODE == 0 ? tape[head] : tape[IN0])) tt |= (1u << x);
            }
            if (!good) continue;
            kept_L++;
            double meanT = 0, maxT = 0;
            for (int x = 0; x < NIN; x++) { meanT += steps[x]; if (steps[x] > maxT) maxT = steps[x]; }
            meanT /= NIN;
            // a program of length 0 has runtime 0; use runtime+1 inside logs
            Beh *b = &beh[tt];
            b->weight += wL; b->count += 1; b->cnt_by_L[L] += 1;
            b->w_meanT += wL * meanT; b->w_maxT += wL * maxT;
            double ktm = L * log2_5 + log2(meanT + 1), ktx = L * log2_5 + log2(maxT + 1);
            if (ktm < b->Kt_mean) b->Kt_mean = ktm;
            if (ktx < b->Kt_max) b->Kt_max = ktx;
            if (L < b->K) {
                b->K = L; b->short_min_meanT = meanT; b->short_min_maxT = maxT;
                for (int i = 0; i < L; i++) b->witness[i] = SYM[prog[i]]; b->witness[L] = 0;
            } else if (L == b->K) {
                if (meanT < b->short_min_meanT) {
                    b->short_min_meanT = meanT;
                    for (int i = 0; i < L; i++) b->witness[i] = SYM[prog[i]]; b->witness[L] = 0;
                }
                if (maxT < b->short_min_maxT) b->short_min_maxT = maxT;
            }
        }
        total_valid_bracket += valid_L; total_kept += kept_L;
        fprintf(stderr, "L=%2d  bracket-valid=%10lld  kept(halting,on-tape)=%10lld\n", L, valid_L, kept_L);
    }
    FILE *f = fopen(out, "w");
    fprintf(f, "tt,weight,K,count,mean_meanT,mean_maxT,Kt_mean,Kt_max,short_min_meanT,short_min_maxT,witness");
    for (int L = 0; L <= Lmax; L++) fprintf(f, ",n%d", L);
    fprintf(f, "\n");
    int nb = 0;
    for (int t = 0; t < 65536; t++) {
        Beh *b = &beh[t];
        if (b->weight <= 0) continue;
        nb++;
        fprintf(f, "%d,%.17g,%d,%.0f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%s", t, b->weight, b->K, b->count,
                b->w_meanT / b->weight, b->w_maxT / b->weight, b->Kt_mean, b->Kt_max,
                b->short_min_meanT, b->short_min_maxT, b->K == 0 ? "e" : b->witness);
        for (int L = 0; L <= Lmax; L++) fprintf(f, ",%.0f", b->cnt_by_L[L]);
        fprintf(f, "\n");
    }
    fclose(f);
    fprintf(stderr, "distinct behaviours: %d   kept programs: %lld\n", nb, total_kept);
    return 0;
}
