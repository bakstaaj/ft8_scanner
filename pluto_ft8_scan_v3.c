/*
 * pluto_ft8_scan.c
 *
 * Local Pluto+/PlutoSDR FT8 scanner.
 *
 * - Uses libiio / AD9361 access style similar to dump1090.c Pluto support.
 * - Captures 15 second FT8 receive windows.
 * - Decimates Pluto IQ to 12 kHz audio-style samples.
 * - Feeds samples to kgoba/ft8_lib.
 * - Prints decoded FT8 text and estimated received frequency.
 * - Supports a 125 MHz HF upconverter such as the Nooelec Ham It Up.
 * - Handles Ctrl-C / SIGTERM and releases the PlutoSDR cleanly.
 *
 * Build requires:
 *   libiio
 *   libm
 *   pthread
 *   kgoba/ft8_lib source tree
 *
 * Example build:
 *   gcc -O3 -Wall -Wextra \
 *     -Ift8_lib/common \
 *     -Ift8_lib/ft8 \
 *     -Ift8_lib/fft \
 *     -Ift8_lib/utils \
 *     pluto_ft8_scan.c \
 *     ft8_lib/common/*.c \
 *     ft8_lib/ft8/*.c \
 *     ft8_lib/fft/*.c \
 *     ft8_lib/utils/*.c \
 *     -o pluto_ft8_scan \
 *     -liio -lm -lpthread
 *
 * Examples:
 *   ./pluto_ft8_scan --scan --verbose
 *   ./pluto_ft8_scan --freq 14074000 --verbose
 *   ./pluto_ft8_scan --scan --ham-it-up --verbose
 *   ./pluto_ft8_scan --freq 14074000 --ham-it-up --gain 45
 *
 * Notes:
 *   FT8 slots start on UTC seconds 00, 15, 30, 45.
 *   This program scans one dial frequency per FT8 window.
 *   HF FT8 is normally USB; tune to the FT8 dial frequency and decode audio tones above it.
 *   With --ham-it-up, the Pluto is tuned to HF_dial + 125 MHz, while printed rx values stay on HF.
 */

#define _POSIX_C_SOURCE 199309L

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <iio.h>

#include "constants.h"
#include "decode.h"
#include "message.h"
#include "monitor.h"

#ifndef ARRAY_LEN
#define ARRAY_LEN(a) ((int)(sizeof(a) / sizeof((a)[0])))
#endif

#define MODES_DEFAULT_RATE       768000
#define MODES_FALLBACK_RATE      1000000
#define MODES_AUDIO_RATE         12000
#define MODES_RF_BW              200000
#define MODES_AUTO_GAIN          -100
#define MODES_MAX_GAIN           70
#define MODES_ASYNC_BUF_NUMBER   12
#define MODES_DATA_LEN           (16 * 16384)

#define FT8_CAPTURE_SECONDS      15
#define FT8_CAPTURE_TRIM_SECONDS 0.20f

#define FT8_MIN_AUDIO_HZ         200.0f
#define FT8_MAX_AUDIO_HZ         3000.0f

#define FT8_MIN_SCORE            10
#define FT8_MAX_CANDIDATES       140
#define FT8_LDPC_ITERATIONS      25
#define FT8_MAX_DECODED          50
#define FT8_FREQ_OSR             2
#define FT8_TIME_OSR             2

#define CALLSIGN_HASHTABLE_SIZE  256

#define MODES_NOTUSED(V) ((void)(V))

struct ft8_frequency {
    const char *name;
    long long dial_hz;
};

static const struct ft8_frequency default_ft8_freqs[] = {
    { "160m",  1840000LL },
    { "80m",   3573000LL },
    { "60m",   5357000LL },
    { "40m",   7074000LL },
    { "30m",  10136000LL },
    { "20m",  14074000LL },
    { "17m",  18100000LL },
    { "15m",  21074000LL },
    { "12m",  24915000LL },
    { "10m",  28074000LL },
    { "6m",   50313000LL }
};

static struct {
    pthread_mutex_t data_mutex;
    pthread_cond_t data_cond;

    int dev_index;
    int gain;
    int enable_agc;
    int verbose;
    volatile sig_atomic_t stop;

    struct iio_context *ctx;
    struct iio_device  *dev;
    struct iio_device  *phy;
    struct iio_channel *rx0_i;
    struct iio_channel *rx0_q;
    struct iio_channel *phy_rx0;
    struct iio_channel *lo_chn;
    struct iio_buffer  *rxbuf;

    long long freq;
    long long sample_rate;

    int scan;
    int cycles;

    int upconverter;
    long long upconverter_offset;
} Modes;

static struct {
    char callsign[12];
    uint32_t hash;
} callsign_hashtable[CALLSIGN_HASHTABLE_SIZE];

static int callsign_hashtable_size = 0;

static void modesCleanup(void);

static long long mstime(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((long long)ts.tv_sec * 1000LL) + (ts.tv_nsec / 1000000LL);
}

static void sigintHandler(int sig) {
    MODES_NOTUSED(sig);
    Modes.stop = 1;

}

static int write_attr_ll(struct iio_channel *chn, const char *attr, long long value) {
    int rc = iio_channel_attr_write_longlong(chn, attr, value);
    if (rc < 0) {
        fprintf(stderr, "iio: failed setting %s=%lld: %s (%d)\n",
                attr, value, strerror(-rc), rc);
    }
    return rc;
}

static int write_attr_str(struct iio_channel *chn, const char *attr, const char *value) {
    int rc = iio_channel_attr_write(chn, attr, value);
    if (rc < 0) {
        fprintf(stderr, "iio: failed setting %s=%s: %s (%d)\n",
                attr, value, strerror(-rc), rc);
    }
    return rc;
}

static void modesInitConfig(void) {
    memset(&Modes, 0, sizeof(Modes));
    Modes.gain = MODES_AUTO_GAIN;
    Modes.enable_agc = 1;
    Modes.dev_index = 0;
    Modes.freq = 14074000LL;
    Modes.sample_rate = MODES_DEFAULT_RATE;
    Modes.scan = 1;
    Modes.cycles = 0;

    Modes.upconverter = 0;
    Modes.upconverter_offset = 125000000LL;   /* Nooelec Ham It Up default LO */
}

static void modesInit(void) {
    pthread_mutex_init(&Modes.data_mutex, NULL);
    pthread_cond_init(&Modes.data_cond, NULL);
}

static void modesInitPLUTOSDR(void) {
    int device_count;

    printf("* Acquiring IIO context\n");

    Modes.ctx = iio_create_default_context();
    if (Modes.ctx == NULL) {
        Modes.ctx = iio_create_network_context("pluto.local");
    }

    if (Modes.ctx == NULL) {
        fprintf(stderr, "No IIO context. Is the Pluto connected or reachable as pluto.local?\n");
        exit(1);
    }

    iio_context_set_timeout(Modes.ctx, 5000);

    device_count = (int)iio_context_get_devices_count(Modes.ctx);
    if (!device_count) {
        fprintf(stderr, "No supported PlutoSDR devices found.\n");
        exit(1);
    }

    if (Modes.verbose) {
        fprintf(stderr, "Found %d IIO device(s)\n", device_count);
    }

    printf("* Acquiring AD9361 streaming device\n");

    Modes.dev = iio_context_find_device(Modes.ctx, "cf-ad9361-lpc");
    if (!Modes.dev) {
        fprintf(stderr, "Could not find cf-ad9361-lpc streaming device\n");
        exit(1);
    }

    Modes.phy = iio_context_find_device(Modes.ctx, "ad9361-phy");
    if (!Modes.phy) {
        fprintf(stderr, "Could not find ad9361-phy device\n");
        exit(1);
    }

    Modes.phy_rx0 = iio_device_find_channel(Modes.phy, "voltage0", false);
    if (!Modes.phy_rx0) {
        fprintf(stderr, "Could not find ad9361-phy voltage0 RX channel\n");
        exit(1);
    }

    Modes.lo_chn = iio_device_find_channel(Modes.phy, "altvoltage0", true);
    if (!Modes.lo_chn) {
        fprintf(stderr, "Could not find ad9361-phy altvoltage0 LO channel\n");
        exit(1);
    }

    printf("* Configuring AD9361 RX\n");

    write_attr_str(Modes.phy_rx0, "rf_port_select", "A_BALANCED");
    write_attr_ll(Modes.phy_rx0, "rf_bandwidth", MODES_RF_BW);

    if (write_attr_ll(Modes.phy_rx0, "sampling_frequency", Modes.sample_rate) < 0) {
        fprintf(stderr, "Retrying sampling_frequency with fallback %d\n", MODES_FALLBACK_RATE);
        Modes.sample_rate = MODES_FALLBACK_RATE;
        if (write_attr_ll(Modes.phy_rx0, "sampling_frequency", Modes.sample_rate) < 0) {
            fprintf(stderr, "Unable to set a valid sample rate.\n");
            exit(1);
        }
    }

    if (Modes.gain == MODES_AUTO_GAIN) {
        Modes.enable_agc = 1;
        write_attr_str(Modes.phy_rx0, "gain_control_mode", "slow_attack");
    } else {
        if (Modes.gain > MODES_MAX_GAIN) {
            Modes.gain = MODES_MAX_GAIN;
        }
        Modes.enable_agc = 0;
        write_attr_str(Modes.phy_rx0, "gain_control_mode", "manual");
        write_attr_ll(Modes.phy_rx0, "hardwaregain", Modes.gain);
    }

    Modes.rx0_i = iio_device_find_channel(Modes.dev, "voltage0", false);
    Modes.rx0_q = iio_device_find_channel(Modes.dev, "voltage1", false);

    if (!Modes.rx0_i || !Modes.rx0_q) {
        fprintf(stderr, "Could not find Pluto RX voltage0/voltage1 streaming channels\n");
        exit(1);
    }

    printf("* Enabling IIO streaming channels\n");
    iio_channel_enable(Modes.rx0_i);
    iio_channel_enable(Modes.rx0_q);

    printf("* Creating non-cyclic IIO RX buffer\n");
    Modes.rxbuf = iio_device_create_buffer(Modes.dev, MODES_DATA_LEN / 2, false);
    if (!Modes.rxbuf) {
        perror("Could not create RX buffer");
        exit(1);
    }
}

static long long getPlutoTuneFrequency(long long hf_dial_hz) {
    if (Modes.upconverter) {
        return hf_dial_hz + Modes.upconverter_offset;
    }

    return hf_dial_hz;
}

static void tunePLUTOSDR(long long hf_dial_hz) {
    int rc;

    long long pluto_tune_hz = getPlutoTuneFrequency(hf_dial_hz);

    Modes.freq = hf_dial_hz;

    if (Modes.verbose) {
        if (Modes.upconverter) {
            printf("Tuning Pluto to %lld Hz for HF dial %lld Hz using upconverter offset %lld Hz\n",
                   pluto_tune_hz,
                   hf_dial_hz,
                   Modes.upconverter_offset);
        } else {
            printf("Tuning Pluto to %lld Hz\n", pluto_tune_hz);
        }
    }

    rc = iio_channel_attr_write_longlong(Modes.lo_chn, "frequency", pluto_tune_hz);
    if (rc < 0) {
        fprintf(stderr, "Failed to tune Pluto frequency %lld Hz: %s (%d)\n",
                pluto_tune_hz, strerror(-rc), rc);
    }

    struct timespec req = { 0, 200000000L };
    nanosleep(&req, NULL);
}

static void waitForFT8Slot(void) {
    while (!Modes.stop) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);

        int sec = (int)(ts.tv_sec % 15);
        long nsec = ts.tv_nsec;

        if (sec == 0 && nsec < 100000000L) {
            return;
        }

        struct timespec nap = { 0, 50000000L };
        nanosleep(&nap, NULL);
    }
}

static int captureAudioWindow(float *audio, int max_audio_samples) {
    const int decim = (int)(Modes.sample_rate / MODES_AUDIO_RATE);
    int out_count = 0;
    int decim_count = 0;
    float acc = 0.0f;

    if (decim <= 0) {
        fprintf(stderr, "Bad decimation factor from sample_rate=%lld\n", Modes.sample_rate);
        return -1;
    }

    if (Modes.sample_rate % MODES_AUDIO_RATE != 0) {
        fprintf(stderr,
                "Warning: sample rate %lld is not an integer multiple of %d; simple decimator will be approximate.\n",
                Modes.sample_rate, MODES_AUDIO_RATE);
    }

    long long start_ms = mstime();

    while (!Modes.stop && out_count < max_audio_samples) {
        ssize_t nbytes = iio_buffer_refill(Modes.rxbuf);
        if (nbytes < 0) {
            if (Modes.stop) {
                break;
            }
            fprintf(stderr, "iio_buffer_refill failed: %s (%zd)\n", strerror((int)-nbytes), nbytes);
            return -1;
        }

        void *p_dat = iio_buffer_first(Modes.rxbuf, Modes.rx0_i);
        void *p_end = iio_buffer_end(Modes.rxbuf);
        ptrdiff_t p_inc = iio_buffer_step(Modes.rxbuf);

        for (; p_dat < p_end && out_count < max_audio_samples; p_dat += p_inc) {
            const int16_t i = ((int16_t *)p_dat)[0];

            float x = (float)i / 32768.0f;

            acc += x;
            decim_count++;

            if (decim_count >= decim) {
                audio[out_count++] = acc / (float)decim_count;
                acc = 0.0f;
                decim_count = 0;
            }
        }
    }

    if (Modes.verbose) {
        long long elapsed = mstime() - start_ms;
        fprintf(stderr, "Captured %d audio samples in %lld ms\n", out_count, elapsed);
    }

    return out_count;
}


static void hashtable_init(void) {
    callsign_hashtable_size = 0;
    memset(callsign_hashtable, 0, sizeof(callsign_hashtable));
}

static void hashtable_cleanup(uint8_t max_age) {
    for (int idx = 0; idx < CALLSIGN_HASHTABLE_SIZE; ++idx) {
        if (callsign_hashtable[idx].callsign[0] != '\0') {
            uint8_t age = (uint8_t)(callsign_hashtable[idx].hash >> 24);

            if (age > max_age) {
                callsign_hashtable[idx].callsign[0] = '\0';
                callsign_hashtable[idx].hash = 0;
                callsign_hashtable_size--;
            } else {
                callsign_hashtable[idx].hash =
                    (((uint32_t)age + 1u) << 24) |
                    (callsign_hashtable[idx].hash & 0x3FFFFFu);
            }
        }
    }
}

static void hashtable_add(const char *callsign, uint32_t hash) {
    uint16_t hash10 = (hash >> 12) & 0x3FFu;
    int idx = (hash10 * 23) % CALLSIGN_HASHTABLE_SIZE;

    while (callsign_hashtable[idx].callsign[0] != '\0') {
        if (((callsign_hashtable[idx].hash & 0x3FFFFFu) == hash) &&
            strcmp(callsign_hashtable[idx].callsign, callsign) == 0) {
            callsign_hashtable[idx].hash &= 0x3FFFFFu;
            return;
        }

        idx = (idx + 1) % CALLSIGN_HASHTABLE_SIZE;
    }

    callsign_hashtable_size++;
    strncpy(callsign_hashtable[idx].callsign, callsign, 11);
    callsign_hashtable[idx].callsign[11] = '\0';
    callsign_hashtable[idx].hash = hash;
}

static bool hashtable_lookup(ftx_callsign_hash_type_t hash_type,
                             uint32_t hash,
                             char *callsign) {
    uint8_t hash_shift =
        (hash_type == FTX_CALLSIGN_HASH_10_BITS) ? 12 :
        (hash_type == FTX_CALLSIGN_HASH_12_BITS) ? 10 : 0;

    uint16_t hash10 = (hash >> (12 - hash_shift)) & 0x3FFu;
    int idx = (hash10 * 23) % CALLSIGN_HASHTABLE_SIZE;

    while (callsign_hashtable[idx].callsign[0] != '\0') {
        if (((callsign_hashtable[idx].hash & 0x3FFFFFu) >> hash_shift) == hash) {
            strcpy(callsign, callsign_hashtable[idx].callsign);
            return true;
        }

        idx = (idx + 1) % CALLSIGN_HASHTABLE_SIZE;
    }

    callsign[0] = '\0';
    return false;
}

static ftx_callsign_hash_interface_t hash_if = {
    .lookup_hash = hashtable_lookup,
    .save_hash = hashtable_add
};

static int decodeFT8Window(const float *audio,
                           int num_samples,
                           long long dial_freq_hz,
                           const char *band_name) {
    monitor_t mon;

    monitor_config_t mon_cfg = {
        .f_min = FT8_MIN_AUDIO_HZ,
        .f_max = FT8_MAX_AUDIO_HZ,
        .sample_rate = MODES_AUDIO_RATE,
        .time_osr = FT8_TIME_OSR,
        .freq_osr = FT8_FREQ_OSR,
        .protocol = FTX_PROTOCOL_FT8
    };

    monitor_init(&mon, &mon_cfg);

    for (int frame_pos = 0;
         !Modes.stop && frame_pos + mon.block_size <= num_samples;
         frame_pos += mon.block_size) {
        monitor_process(&mon, audio + frame_pos);
    }

    if (Modes.stop) {
        monitor_free(&mon);
        return 0;
    }

    const ftx_waterfall_t *wf = &mon.wf;

    ftx_candidate_t candidate_list[FT8_MAX_CANDIDATES];
    int num_candidates = ftx_find_candidates(
        wf,
        FT8_MAX_CANDIDATES,
        candidate_list,
        FT8_MIN_SCORE
    );

    int num_decoded = 0;
    ftx_message_t decoded[FT8_MAX_DECODED];
    ftx_message_t *decoded_hashtable[FT8_MAX_DECODED];

    for (int i = 0; i < FT8_MAX_DECODED; ++i) {
        decoded_hashtable[i] = NULL;
    }

    time_t now = time(NULL);
    struct tm utc;
    gmtime_r(&now, &utc);

    for (int idx = 0; !Modes.stop && idx < num_candidates; ++idx) {
        const ftx_candidate_t *cand = &candidate_list[idx];

        float audio_hz =
            (mon.min_bin + cand->freq_offset + ((float)cand->freq_sub / wf->freq_osr))
            / mon.symbol_period;

        float time_sec =
            (cand->time_offset + ((float)cand->time_sub / wf->time_osr))
            * mon.symbol_period;

        ftx_message_t message;
        ftx_decode_status_t status;

        if (!ftx_decode_candidate(wf, cand, FT8_LDPC_ITERATIONS, &message, &status)) {
            continue;
        }

        int hash_idx = message.hash % FT8_MAX_DECODED;
        bool found_empty = false;
        bool found_duplicate = false;

        do {
            if (decoded_hashtable[hash_idx] == NULL) {
                found_empty = true;
            } else if ((decoded_hashtable[hash_idx]->hash == message.hash) &&
                       memcmp(decoded_hashtable[hash_idx]->payload,
                              message.payload,
                              sizeof(message.payload)) == 0) {
                found_duplicate = true;
            } else {
                hash_idx = (hash_idx + 1) % FT8_MAX_DECODED;
            }
        } while (!found_empty && !found_duplicate);

        if (found_duplicate) {
            continue;
        }

        memcpy(&decoded[hash_idx], &message, sizeof(message));
        decoded_hashtable[hash_idx] = &decoded[hash_idx];
        num_decoded++;

        char text[FTX_MAX_MESSAGE_LENGTH];
        ftx_message_offsets_t offsets;

        ftx_message_rc_t unpack_status =
            ftx_message_decode(&message, &hash_if, text, &offsets);

        if (unpack_status != FTX_MESSAGE_RC_OK) {
            snprintf(text, sizeof(text), "UNPACK_ERROR_%d", (int)unpack_status);
        }

        long long received_hz = dial_freq_hz + (long long)llroundf(audio_hz);

        /* cand->score is not true SNR. This is only a rough score-based display. */
        float snr_est = cand->score * 0.5f;

        printf("%02d:%02d:%02dZ  %-4s  dial=%lld Hz  rx=%lld Hz  audio=%7.1f Hz  dt=%4.2f  score=%d  snr~%+05.1f  %s\n",
               utc.tm_hour,
               utc.tm_min,
               utc.tm_sec,
               band_name ? band_name : "user",
               dial_freq_hz,
               received_hz,
               audio_hz,
               time_sec,
               cand->score,
               snr_est,
               text);

        fflush(stdout);
    }

    if (Modes.verbose) {
        fprintf(stderr,
                "Decode summary: band=%s hf_dial=%lld pluto_tuned=%lld candidates=%d decoded=%d max_mag=%.1f dB hash_size=%d\n",
                band_name ? band_name : "user",
                dial_freq_hz,
                getPlutoTuneFrequency(dial_freq_hz),
                num_candidates,
                num_decoded,
                mon.max_mag,
                callsign_hashtable_size);
    }

    hashtable_cleanup(10);
    monitor_free(&mon);

    return num_decoded;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "Options:\n"
        "  --scan                  Scan common FT8 dial frequencies, default\n"
        "  --freq HZ               Decode one HF dial frequency only\n"
        "  --gain DB               Manual RX gain, 0..70. Default is AGC\n"
        "  --agc                   Use slow_attack AGC\n"
        "  --upconverter           Enable HF upconverter mode\n"
        "  --upconverter-offset HZ Upconverter LO offset. Default 125000000\n"
        "  --ham-it-up             Shortcut for --upconverter --upconverter-offset 125000000\n"
        "  --cycles N              Number of scan cycles. 0 = forever\n"
        "  --verbose               Print tuning/capture/decode diagnostics\n"
        "  --help                  Show this help\n\n"
        "Examples:\n"
        "  %s --scan --verbose\n"
        "  %s --scan --ham-it-up --verbose\n"
        "  %s --freq 14074000 --ham-it-up --gain 45\n"
        "  %s --freq 7074000 --upconverter --upconverter-offset 125000000\n",
        prog, prog, prog, prog, prog);
}

static void parseArgs(int argc, char **argv) {
    for (int j = 1; j < argc; j++) {
        if (!strcmp(argv[j], "--scan")) {
            Modes.scan = 1;
        } else if (!strcmp(argv[j], "--freq") && j + 1 < argc) {
            Modes.freq = atoll(argv[++j]);
            Modes.scan = 0;
        } else if (!strcmp(argv[j], "--gain") && j + 1 < argc) {
            Modes.gain = atoi(argv[++j]);
            Modes.enable_agc = 0;
        } else if (!strcmp(argv[j], "--agc")) {
            Modes.gain = MODES_AUTO_GAIN;
            Modes.enable_agc = 1;
        } else if (!strcmp(argv[j], "--upconverter")) {
            Modes.upconverter = 1;
        } else if (!strcmp(argv[j], "--upconverter-offset") && j + 1 < argc) {
            Modes.upconverter = 1;
            Modes.upconverter_offset = atoll(argv[++j]);
        } else if (!strcmp(argv[j], "--ham-it-up")) {
            Modes.upconverter = 1;
            Modes.upconverter_offset = 125000000LL;
        } else if (!strcmp(argv[j], "--cycles") && j + 1 < argc) {
            Modes.cycles = atoi(argv[++j]);
        } else if (!strcmp(argv[j], "--verbose")) {
            Modes.verbose = 1;
        } else if (!strcmp(argv[j], "--help")) {
            usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "Unknown or incomplete option: %s\n", argv[j]);
            usage(argv[0]);
            exit(1);
        }
    }
}

static void modesCleanup(void) {
    fprintf(stderr, "Releasing PlutoSDR resources...\n");

    if (Modes.rxbuf) {
        iio_buffer_destroy(Modes.rxbuf);
        Modes.rxbuf = NULL;
    }

    if (Modes.rx0_i) {
        iio_channel_disable(Modes.rx0_i);
        Modes.rx0_i = NULL;
    }

    if (Modes.rx0_q) {
        iio_channel_disable(Modes.rx0_q);
        Modes.rx0_q = NULL;
    }

    if (Modes.ctx) {
        iio_context_destroy(Modes.ctx);
        Modes.ctx = NULL;
    }

    pthread_cond_destroy(&Modes.data_cond);
    pthread_mutex_destroy(&Modes.data_mutex);

    fprintf(stderr, "PlutoSDR resources released.\n");
}

int main(int argc, char **argv) {
    modesInitConfig();
    parseArgs(argc, argv);
    modesInit();

    signal(SIGINT, sigintHandler);
    signal(SIGTERM, sigintHandler);

    modesInitPLUTOSDR();
    hashtable_init();

    const int full_samples = FT8_CAPTURE_SECONDS * MODES_AUDIO_RATE;
    const int use_samples =
        (int)((FT8_CAPTURE_SECONDS - FT8_CAPTURE_TRIM_SECONDS) * MODES_AUDIO_RATE);

    float *audio = (float *)calloc(full_samples, sizeof(float));
    if (!audio) {
        fprintf(stderr, "Out of memory allocating FT8 audio buffer\n");
        modesCleanup();
        return 1;
    }

    int completed_cycles = 0;

    while (!Modes.stop) {
        if (Modes.scan) {
            for (int i = 0; i < ARRAY_LEN(default_ft8_freqs) && !Modes.stop; i++) {
                const struct ft8_frequency *f = &default_ft8_freqs[i];

                tunePLUTOSDR(f->dial_hz);

                if (Modes.verbose) {
                    fprintf(stderr, "Waiting for FT8 slot: %s %lld Hz\n",
                            f->name, f->dial_hz);
                }

                waitForFT8Slot();

                if (Modes.stop) {
                    break;
                }

                int got = captureAudioWindow(audio, use_samples);
                if (got > 0 && !Modes.stop) {
                    decodeFT8Window(audio, got, f->dial_hz, f->name);
                }
            }

            completed_cycles++;
            if (Modes.cycles > 0 && completed_cycles >= Modes.cycles) {
                break;
            }
        } else {
            tunePLUTOSDR(Modes.freq);

            if (Modes.verbose) {
                fprintf(stderr, "Waiting for FT8 slot: %lld Hz\n", Modes.freq);
            }

            waitForFT8Slot();

            if (Modes.stop) {
                break;
            }

            int got = captureAudioWindow(audio, use_samples);
            if (got > 0 && !Modes.stop) {
                decodeFT8Window(audio, got, Modes.freq, "user");
            }

            completed_cycles++;
            if (Modes.cycles > 0 && completed_cycles >= Modes.cycles) {
                break;
            }
        }
    }

    if (Modes.stop) {
        fprintf(stderr, "Ctrl-C/SIGTERM received, shutting down cleanly.\n");
    }

    free(audio);
    modesCleanup();

    fprintf(stderr, "Clean exit.\n");
    return 0;
}
