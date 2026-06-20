/* libspeedhack.so — LD_PRELOAD time-scaling shim. The Linux equivalent of NitroGen's Windows
 * `xspeedhack`: it lets an external controller freeze/slow/run a game by SCALING the monotonic
 * clock the game uses for timing, WITHOUT the game's source. doukutsu-rs (Rust) and most
 * SDL/C/C++ games read time via glibc clock_gettime(CLOCK_MONOTONIC); we override that libc
 * symbol so the game perceives time passing at `scale` x real rate:
 *     scale = 0.0  -> game clock FROZEN (pause; inference latency invisible)
 *     scale = 1.0  -> normal speed
 *     scale = k    -> k x speed
 *
 * Continuity: scaled = anchor_scaled + (real_now - anchor_real) * scale, re-anchored whenever
 * the controller changes the scale, so there is never a time jump (matching xspeedhack's
 * behaviour). One anchor per clock id (monotonic family only); other clocks pass through.
 *
 * Control channel: an mmap'd file at $SPEEDHACK_CTRL whose first 8 bytes are a double `scale`,
 * written live by the Python controller (nitrogen/eval/speedhack.py). If unset/missing, the
 * shim is a transparent passthrough (safe).
 *
 * Build: gcc -shared -fPIC -O2 libspeedhack.c -o libspeedhack.so -ldl -lpthread
 * Use:   SPEEDHACK_CTRL=/path/ctl LD_PRELOAD=/path/libspeedhack.so ./game
 *
 * Caveats (shared with xspeedhack): only intercepts libc clock_gettime — does NOT affect Go
 * programs (raw syscalls) or statically-linked musl binaries; a game timing off rdtsc or the
 * audio clock won't be scaled. Covers the large majority (C/C++/Rust/SDL).
 */
#define _GNU_SOURCE
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/mman.h>

typedef int (*cgt_t)(clockid_t, struct timespec *);

static cgt_t real_cgt = (cgt_t)0;
static volatile double *ctrl = (double *)0;  /* mmap'd scale, written by the controller */
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

#define NCLK 16
static int64_t anc_real[NCLK];
static int64_t anc_scaled[NCLK];
static double  cur_scale[NCLK];
static int     clk_init[NCLK];
static int     g_init = 0;

static int64_t ts_ns(const struct timespec *t) {
    return (int64_t)t->tv_sec * 1000000000LL + (int64_t)t->tv_nsec;
}
static void ns_ts(int64_t ns, struct timespec *t) {
    t->tv_sec  = (time_t)(ns / 1000000000LL);
    t->tv_nsec = (long)(ns % 1000000000LL);
}

static int is_scaled(clockid_t c) {
    if (c == CLOCK_MONOTONIC) return 1;
#ifdef CLOCK_MONOTONIC_RAW
    if (c == CLOCK_MONOTONIC_RAW) return 1;
#endif
#ifdef CLOCK_BOOTTIME
    if (c == CLOCK_BOOTTIME) return 1;
#endif
    return 0;
}

static void ensure_init(void) {
    if (g_init) return;
    real_cgt = (cgt_t)dlsym(RTLD_NEXT, "clock_gettime");
    const char *path = getenv("SPEEDHACK_CTRL");
    if (path) {
        int fd = open(path, O_RDWR);
        if (fd >= 0) {
            void *m = mmap((void *)0, sizeof(double), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (m != MAP_FAILED) ctrl = (double *)m;
            close(fd);
        }
    }
    g_init = 1;
}

int clock_gettime(clockid_t clk, struct timespec *tp) {
    if (!g_init) ensure_init();
    if (!real_cgt) {            /* extreme fallback */
        tp->tv_sec = 0; tp->tv_nsec = 0; return 0;
    }
    if (!ctrl || !is_scaled(clk) || clk < 0 || clk >= NCLK)
        return real_cgt(clk, tp);

    struct timespec t;
    if (real_cgt(clk, &t) != 0) return -1;
    int64_t real = ts_ns(&t);

    pthread_mutex_lock(&lock);
    double target = *ctrl;
    if (!clk_init[clk]) {
        anc_real[clk] = real; anc_scaled[clk] = real;
        cur_scale[clk] = target; clk_init[clk] = 1;
    }
    if (target != cur_scale[clk]) {           /* re-anchor on scale change (no jump) */
        anc_scaled[clk] += (int64_t)((double)(real - anc_real[clk]) * cur_scale[clk]);
        anc_real[clk] = real;
        cur_scale[clk] = target;
    }
    int64_t scaled = anc_scaled[clk] + (int64_t)((double)(real - anc_real[clk]) * cur_scale[clk]);
    pthread_mutex_unlock(&lock);

    ns_ts(scaled, tp);
    return 0;
}
