# Heat-adaptive render scale for builds with a frame step state page: DM_HEAT_SCALE=<fair>,<serious>
import sys
p = sys.argv[1]; s = open(p).read()
anchor = '#define THERMAL_CALM_NS 30000000000ull\n'
assert s.count(anchor) == 1, 'thermal cap moved upstream'
code = r'''// Heat-adaptive render scale (DM_HEAT_SCALE=<fair>,<serious>, each 0.5 to 1.0): while iOS reports that thermal state the 3D views' scale (state +0x10, read by the
// game's stubs every frame) is lowered, so the GPU makes less heat and the 30 fps cap above comes later or not at all. Down at once; back up after THERMAL_CALM_NS at a lower state
static float hs_fair = -1, hs_serious = -1, hs_base = -1, hs_cur = -1; static uint64_t hs_calm;
static int hs_on(void) {
    if (hs_fair < 0) {
        hs_fair = 0; const char *e = getenv("DM_HEAT_SCALE"); float a, b;
        if (e && sscanf(e, "%f,%f", &a, &b) == 2 && a >= 0.5f && a <= 1.0f && b >= 0.5f && b <= 1.0f) { hs_fair = a; hs_serious = b; LOG("[nvn] heat-adaptive render scale: fair %.2f, serious %.2f (DM_HEAT_SCALE)", a, b); }
        else if (e) LOG("[RED] DM_HEAT_SCALE=%s: two scales from 0.5 to 1.0, <fair>,<serious>", e);
    }
    return hs_fair > 0 && dm_fps_state != NULL;
}
static void hs_tick(int th, uint64_t now) {
    if (!hs_on() || th < 0) return;
    if (hs_base < 0) { memcpy(&hs_base, dm_fps_state + 0x10, sizeof hs_base); if (!(hs_base >= 0.5f && hs_base <= 1.0f)) hs_base = 1.0f; hs_cur = hs_base; }
    float want = th >= 2 ? hs_serious : th == 1 ? hs_fair : hs_base;
    if (want > hs_base) want = hs_base;
    if (want == hs_cur) { hs_calm = 0; return; }
    if (want > hs_cur) { if (!hs_calm) { hs_calm = now; return; } if (now - hs_calm < THERMAL_CALM_NS) return; }
    hs_calm = 0; hs_cur = want;
    uint32_t bits; memcpy(&bits, &want, sizeof bits); __atomic_store_n((uint32_t *)(dm_fps_state + 0x10), bits, __ATOMIC_RELAXED);
    LOG("[nvn] thermal state %d: render scale %.2f (DM_HEAT_SCALE)", th, want);
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
s = s.replace(old_on, '''    if (on || hs_on()) {
        uint64_t now = now_ns();
        if (now - last >= 1000000000ull) {   // the thermal state is read once a second
            last = now;
            int th = dm_thermal_now();
            hs_tick(th, now);
            if (!on) {} else if (th >= 2)''', 1)
open(p, 'w').write(s); print('heat-adaptive render scale added')
