# Host-side render scale control for builds with a frame step state page (the 3D views' scale at state +0x10 is read by the game's stubs every frame):
#   DM_HEAT_SCALE=<fair>,<serious>  ceiling while iOS reports that thermal state (down at once, back up after THERMAL_CALM_NS at a lower state)
#   DM_DYN_SCALE=<floor>            at the 60 fps cap: 3+ late presents in a second step the scale down 0.05 (not below floor); 5 s without one step it back up 0.05
import sys
p = sys.argv[1]; s = open(p).read()
anchor = '#define THERMAL_CALM_NS 30000000000ull\n'
assert s.count(anchor) == 1, 'thermal cap moved upstream'
code = r'''static float hs_fair = -1, hs_serious = -1, hs_base = -1, hs_cur = -1, df_floor = -1, df_scale = -1, hs_written = -1; static uint64_t hs_calm;
static uint64_t df_last, df_win, df_good; static int df_late, df_frames; static pthread_mutex_t hs_mu = PTHREAD_MUTEX_INITIALIZER;   // the scale state, shared by the thermal sample and the present thread
static void hs_env(void) {
    if (hs_fair >= 0) return;
    hs_fair = 0; df_floor = 0; float a, b; const char *e = getenv("DM_HEAT_SCALE");
    if (e && sscanf(e, "%f,%f", &a, &b) == 2 && a >= 0.5f && a <= 1.0f && b >= 0.5f && b <= 1.0f) { hs_fair = a; hs_serious = b; LOG("[nvn] heat-adaptive render scale: fair %.2f, serious %.2f (DM_HEAT_SCALE)", a, b); }
    else if (e) LOG("[RED] DM_HEAT_SCALE=%s: two scales from 0.5 to 1.0, <fair>,<serious>", e);
    e = getenv("DM_DYN_SCALE");
    if (e && sscanf(e, "%f", &a) == 1 && a >= 0.5f && a <= 1.0f) { df_floor = a; LOG("[nvn] 60 fps render scale: down 0.05 a second while presents run late, not below %.2f (DM_DYN_SCALE)", a); }
    else if (e) LOG("[RED] DM_DYN_SCALE=%s: a floor from 0.5 to 1.0", e);
}
static int hs_on(void) { hs_env(); return hs_fair > 0 && dm_fps_state != NULL; }
static int df_on(void) { hs_env(); return df_floor > 0 && dm_fps_state != NULL; }
static void hs_init(void) {
    if (hs_base >= 0) return;
    memcpy(&hs_base, dm_fps_state + 0x10, sizeof hs_base); if (!(hs_base >= 0.5f && hs_base <= 1.0f)) hs_base = 1.0f;
    hs_cur = hs_base; df_scale = hs_base; hs_written = hs_base;
}
static void hs_write(const char *why, int th) {   // the scale shown: the lower of the heat ceiling and the 60 fps controller
    float want = hs_cur; if (df_on() && df_scale < want) want = df_scale;
    if (want == hs_written) return;
    hs_written = want; uint32_t bits; memcpy(&bits, &want, sizeof bits); __atomic_store_n((uint32_t *)(dm_fps_state + 0x10), bits, __ATOMIC_RELAXED);
    LOG("[nvn] render scale %.2f (%s, thermal state %d)", want, why, th);
}
static int hs_th = 0;
static void hs_tick_locked(int th, uint64_t now);
static void hs_tick(int th, uint64_t now) {   // once a second, under the thermal mutex
    hs_th = th < 0 ? 0 : th;
    if (!hs_on() || th < 0) return;
    pthread_mutex_lock(&hs_mu); hs_tick_locked(th, now); pthread_mutex_unlock(&hs_mu);
}
static void hs_tick_locked(int th, uint64_t now) {
    hs_init();
    float want = th >= 2 ? hs_serious : th == 1 ? hs_fair : hs_base;
    if (want > hs_base) want = hs_base;
    if (want == hs_cur) { hs_calm = 0; return; }
    if (want > hs_cur) { if (!hs_calm) { hs_calm = now; return; } if (now - hs_calm < THERMAL_CALM_NS) return; }
    hs_calm = 0; hs_cur = want; hs_write("DM_HEAT_SCALE", th);
}
static void hs_frame(uint64_t now, int cap) {   // each present, on the present thread
    if (!df_on()) return;
    pthread_mutex_lock(&hs_mu);
    hs_init();
    if (cap != 60) { df_last = 0; df_win = 0; df_late = df_frames = 0; df_good = 0; pthread_mutex_unlock(&hs_mu); return; }   // at 30 (movie, thermal cap) it waits
    if (df_last) { uint64_t d = now - df_last; if (d < 100000000ull) { df_frames++; if (d > 22000000ull) df_late++; } }   // longer gaps: loading, menus, the app in the background
    df_last = now;
    if (!df_win) df_win = now;
    if (now - df_win >= 1000000000ull) {
        float s = df_scale;
        if (df_late >= 3 && df_frames >= 20) { s -= 0.05f; if (s < df_floor) s = df_floor; df_good = 0; }
        else if (df_late == 0 && df_frames >= 40) { if (!df_good) df_good = now; else if (now - df_good >= 5000000000ull) { s += 0.05f; if (s > hs_base) s = hs_base; df_good = now; } }
        else df_good = 0;
        s = (float)(int)(s * 20.0f + 0.5f) / 20.0f; if (s > hs_base) s = hs_base; if (s < df_floor) s = df_floor;
        df_scale = s; df_win = now; df_late = df_frames = 0;
        hs_write("DM_DYN_SCALE", hs_th);
    }
    pthread_mutex_unlock(&hs_mu);
}
'''
s = s.replace(anchor, anchor + code, 1)
old_on = '''    if (on) {
        uint64_t now = now_ns();
        if (now - last >= 1000000000ull) {   // the thermal state is read once a second
            last = now;
            int th = dm_thermal_now();
            if (th >= 2)'''
assert s.count(old_on) == 1, 'thermal sample block moved upstream'
s = s.replace(old_on, '''    if (on || hs_on() || df_on()) {
        uint64_t now = now_ns();
        if (now - last >= 1000000000ull) {   // the thermal state is read once a second
            last = now;
            int th = dm_thermal_now();
            hs_tick(th, now);
            if (!on) {} else if (th >= 2)''', 1)
pres = '''    int c = fps_step_cap();
    if (dm_fps_state) { static int was;'''
assert s.count(pres) == 1, 'present path moved upstream'
s = s.replace(pres, '''    int c = fps_step_cap();
    hs_frame(now_ns(), c);
    if (dm_fps_state) { static int was;''', 1)
open(p, 'w').write(s); print('host render scale control added (DM_HEAT_SCALE, DM_DYN_SCALE)')
