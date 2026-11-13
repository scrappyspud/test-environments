#!/usr/bin/env python3
# lite.py <nxnative checkout>: lighter-scene switches for Breath of the Wild 1.9.0 (main cd57b23f...), read from env.txt when the game starts:
#   DM_LITE_DIST=<0.5..1>   placed actors' traverse distance x this factor (trees, rocks, buildings, enemies appear nearer; 1 = the game's own)
#   DM_LITE_GRASS=<0.4..1>  grass fade distance x this factor (both grass layers fade out nearer; 1 = the game's own)
# Sites and their meaning from Savage Wind Recomp's research (MIT, github.com/Revivedx/SavageWindRecomp: tools/patches.py _traverse_scale_default,
# _grass_fade; runtime/draw_distance.c), where 0.6 / 0.75 are its tested "Potato" values. The stubs read two floats the host writes into the frame
# step state page (+0x100, +0x104) before the game runs; 1.0 there gives the game's exact behaviour.
# Each stub is placed only when the code around its site (+-1 KB) never names x16 / w16 and, for the grass, a free v16..v31: otherwise it is left out
# (a [lite] line says so) and the switch does nothing.
import sys, os
root = sys.argv[1]

P = os.path.join(root, 'dm/games/01007EF00011E000/patches.py')
s = open(P).read()
a = "FRAME_STEP = {'cd57b23fa4bbad65803d9788c01821ee': frame_step}"
assert s.count(a) == 1, 'FRAME_STEP line moved upstream'
s = s.replace(a, r'''
def _lite_free(lo, hi, names):
    # True when no instruction in [lo, hi) of the original text names any of these registers
    import re, __main__
    from capstone import Cs, CS_ARCH_ARM64, CS_MODE_ARM
    md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    rx = re.compile(r'\b(' + '|'.join(names) + r')\b')
    t = __main__.ORIG_TEXT
    for ins in md.disasm(bytes(t[max(lo, 0):hi]), max(lo, 0)):
        if rx.search(ins.op_str): return False
    return True

def _lite(ctx):
    import struct, __main__
    t = __main__.ORIG_TEXT
    word = lambda off: struct.unpack_from('<I', t, off)[0]
    near = lambda off, names: _lite_free(off - 0x400, off + 0x400, names)
    # Draw distance: the placement's traverse-distance scale without the debug manager (released game), 'fmov s1, #1.0' -> the host's factor
    S = 0x10a4694
    if word(S) == 0x1e2e1001 and near(S, ['x16', 'w16']):
        ctx.site(S, '01102e1e'); st = ctx.here()
        ctx.emit(ADRP(16, ctx.here(), ctx.state)); ctx.emit(0xbd400000 | (0x100 // 4) << 10 | 16 << 5 | 1)   # ldr s1, [x16, #0x100]
        ctx.emit(B(ctx.here(), S + 4))
        ctx.put(S, [B(S, st)], 'lite: draw distance factor (state +0x100)')
        print('[lite] draw distance stub placed')
    else: print('[lite] draw distance stub left out: site %#x is %#x or x16 is used near it' % (S, word(S)))
    # Grass: each layer's store of the camera scale its fades are measured with, 'str s0, [xB, #off]' -> stores s0 / factor (a larger scale fades nearer: 0.6 fades at 0.6 of the distance)
    for S, base, off in ((0x13a85d0, 27, 0x240), (0x13baa5c, 24, 0x4)):
        want = 0xbd000000 | (off // 4) << 10 | base << 5
        fr = next((n for n in range(16, 32) if near(S, [p + str(n) for p in 'bhsdqv'])), None)
        if word(S) != want or fr is None or not near(S, ['x16', 'w16']):
            print('[lite] grass stub at %#x left out (site %#x, free fp register %s)' % (S, word(S), fr)); continue
        ctx.site(S, struct.pack('<I', want).hex()); st = ctx.here()
        ctx.emit(ADRP(16, ctx.here(), ctx.state)); ctx.emit(0xbd400000 | (0x104 // 4) << 10 | 16 << 5 | fr)   # ldr sF, [x16, #0x104]
        ctx.emit(0x1e201800 | fr << 16 | 0 << 5 | fr)                                                        # fdiv sF, s0, sF
        ctx.emit(0xbd000000 | (off // 4) << 10 | base << 5 | fr)                                             # str sF, [xB, #off]
        ctx.emit(B(ctx.here(), S + 4))
        ctx.put(S, [B(S, st)], 'lite: grass fade factor (state +0x104)')
        print('[lite] grass stub placed at %#x (scratch s%d)' % (S, fr))

def _frame_step_lite(ctx):
    frame_step(ctx); _lite(ctx)

FRAME_STEP = {'cd57b23fa4bbad65803d9788c01821ee': _frame_step_lite}''')
assert 'from nso_asm import' in s and ' B' in s.split('from nso_asm import', 1)[1].split('\n', 1)[0] and 'ADRP' in s.split('from nso_asm import', 1)[1].split('\n', 1)[0]
open(P, 'w').write(s)

L = os.path.join(root, 'dm/host/loader.c')
s = open(L).read()
a = 'memcpy(dm_fps_state + 0x10, &sc, sizeof sc);'
assert s.count(a) == 1, 'render scale write moved upstream'
s = s.replace(a, a + r'''
    {   // lite.py: Breath of the Wild's draw distance and grass fade factors (state +0x100, +0x104; 1.0 = the game's own), DM_LITE_DIST / DM_LITE_GRASS
        const char *n[2] = {"DM_LITE_DIST", "DM_LITE_GRASS"}; const float lo[2] = {0.5f, 0.4f};
        for (int i = 0; i < 2; i++) {
            float v = 1.0f; const char *x = getenv(n[i]);
            if (x) { float w = strtof(x, NULL); if (w >= lo[i] && w <= 1.0f) { v = w; LOG("%s=%.2f: %s", n[i], v, i ? "grass fades out nearer" : "placed actors appear nearer"); }
                     else LOG("[RED] %s=%s: must be %.1f to 1.0; the game's own", n[i], x, lo[i]); }
            memcpy(dm_fps_state + 0x100 + 4 * i, &v, sizeof v);
        }
    }''')
open(L, 'w').write(s)
print('lite switches added (DM_LITE_DIST, DM_LITE_GRASS)')
