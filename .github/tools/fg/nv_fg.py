#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# nv_fg.py <nvn_mtl.m>: hooks frame generation (fg_mtl.m, built in with FG_EMBEDDED) into the display-link present path.
# With DM_FG=1 in env.txt and a game frame step of 30 fps or slower (pq item period > 25 ms), the tick that takes a new
# frame N shows the generated frame between N-1 and N, and the next tick (one the present interval would otherwise hold)
# shows N itself: 60 pictures a second from 30 game frames, one refresh (16.7 ms) later than without.
# Ordering: the Metal 4 queue signals an event after the frame's rendering, a classic queue waits for it, copies the frame
# in, runs the chain and signals again, and the Metal 4 queue waits for that before the present that samples the output.
import sys
p = sys.argv[1]; s = open(p).read()

def once(old, new):
    global s
    assert s.count(old) == 1, 'anchor moved upstream: ' + old[:80]
    s = s.replace(old, new, 1)

# 1. The generator and its state, ahead of present_now (which takes the override)
once('static double g_link_it_tin;', r'''#define FG_EMBEDDED 1
#include "fg_mtl.inc"
static __strong id<MTLTexture> g_fg_override;   // present_now shows this instead of the frame's own texture (the generated picture)
static fg_ctx *g_fg; static int g_fg_on = -1; static float g_fg_flow = 1.0f;
static __strong id<MTLCommandQueue> g_fg_q; static __strong id<MTLSharedEvent> g_fg_ev; static uint64_t g_fg_v;
static __strong id<MTLTexture> g_fg_ring[4]; static int g_fg_ri; static uint64_t g_fg_n;   // two pairs, used in turn
static int g_fg_quality, g_fg_dips = 1; static double g_fg_prev_period = 1.0 / 30;
static double g_fg_gpu[300]; static int g_fg_gn; static pthread_mutex_t g_fg_mu = PTHREAD_MUTEX_INITIALIZER;
// A check that the generated pictures really sit between the real ones: 64x48 luma samples of each real frame and each generated
// picture, compared on the CPU when the command buffer completes; logged every 300 generated pictures
static __strong id<MTLComputePipelineState> g_fgp_ps; static __strong id<MTLBuffer> g_fgp_prev; static int g_fgp_bad;
static double g_fgp_gp, g_fgp_gn, g_fgp_pn; static int g_fgp_n;
static id<MTLBuffer> fgp_sample(id<MTLCommandBuffer> cb, id<MTLTexture> t, NSUInteger x, NSUInteger y, NSUInteger w, NSUInteger h) {
    if (!g_fgp_ps && !g_fgp_bad) {
        NSString *src = @"#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void fgp(texture2d<float, access::read> t [[texture(0)]], device float *o [[buffer(0)]], constant uint4 &r [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {\n"
            "  if (g.x >= 64 || g.y >= 48) return; uint x = r.x + (g.x * 2 + 1) * r.z / 128, y = r.y + (g.y * 2 + 1) * r.w / 96;\n"
            "  float4 c = t.read(uint2(x, y)); o[g.y * 64 + g.x] = dot(c.rgb, float3(0.299, 0.587, 0.114)) * 255.0; }\n";
        NSError *e = nil; id<MTLLibrary> l = [g_dev newLibraryWithSource:src options:nil error:&e];
        id<MTLFunction> f = [l newFunctionWithName:@"fgp"]; g_fgp_ps = f ? [g_dev newComputePipelineStateWithFunction:f error:&e] : nil;
        if (!g_fgp_ps) { g_fgp_bad = 1; MLOG("[RED] frame generation check: %s", e.description.UTF8String); return nil; }
    }
    if (!g_fgp_ps) return nil;
    id<MTLBuffer> o = [g_dev newBufferWithLength:64 * 48 * sizeof(float) options:MTLResourceStorageModeShared];
    uint32_t r[4] = {(uint32_t)x, (uint32_t)y, (uint32_t)w, (uint32_t)h};
    id<MTLComputeCommandEncoder> c = [cb computeCommandEncoder]; [c setComputePipelineState:g_fgp_ps];
    [c setTexture:t atIndex:0]; [c setBuffer:o offset:0 atIndex:0]; [c setBytes:r length:sizeof r atIndex:1];
    [c dispatchThreads:MTLSizeMake(64, 48, 1) threadsPerThreadgroup:MTLSizeMake(16, 16, 1)]; [c endEncoding];
    return o;
}
static double fgp_mad(const float *a, const float *b) { double s = 0; for (int i = 0; i < 64 * 48; i++) s += fabsf(a[i] - b[i]); return s / (64 * 48); }
static double g_fg_ema; static uint64_t g_fg_last; static int g_fg_paused;   // g_fg_ema: how late frames come, against the time the frame before was meant to stay up
// The generated frame between the previous frame given here and this one, or nil (off, first frames, unsupported picture, paused).
// Only the part of the picture the game shows (its crop) goes through the chain; the generated picture is that part alone, and
// out_crop is where it sits in the returned texture. While the game's frames come later than their own step (the GPU is the limit,
// e.g. when hot), generation pauses: its GPU time would only slow the game further. A frame the game shows for three refreshes
// (its own drop to 20 fps) gets two pictures, a third and two thirds of the way: *second is the later one (DM_FG_DIPS=0: one picture).
static id<MTLTexture> fg_make(const tex_t *t, double period, const int crop[4], int out_crop[4], __strong id<MTLTexture> *second) {
    *second = nil;
    if (g_fg_on < 0) {
        const char *e = getenv("DM_FG"), *f = getenv("DM_FG_FLOW"), *q = getenv("DM_FG_QUALITY"), *d = getenv("DM_FG_DIPS"); g_fg_on = e && atoi(e) > 0;
        g_fg_quality = q && atoi(q) > 0; g_fg_dips = !(d && !strcmp(d, "0"));
        if (f) { float v = (float)atof(f); if (v >= 0.25f && v <= 1.0f) g_fg_flow = v; else MLOG("[RED] DM_FG_FLOW=%s: a flow scale from 0.25 to 1.0", f); }
        if (g_fg_on) MLOG("frame generation (DM_FG): on while the frame step is 30 fps or slower, one generated picture between each pair of frames%s, %s chain (DM_FG_QUALITY), flow scale %.2f (DM_FG_FLOW)",
                          g_fg_dips ? " and two where the game shows a frame for three refreshes (DM_FG_DIPS)" : "", g_fg_quality ? "quality" : "performance", g_fg_flow);
    }
    if (!g_fg_on || !g_q4) return nil;
    uint64_t tn = now_ns(); double want = g_fg_prev_period * 1e3, dt = g_fg_last ? (tn - g_fg_last) / 1e6 : want; g_fg_last = tn; g_fg_prev_period = period;
    // Pause only when frames keep coming late (12 of the last 16), not for a single hitch such as a loading spike or an autosave
    static uint16_t late_bits; static uint32_t on_time_run;
    if (dt < 200) { int late = dt - want > 4.0; late_bits = (uint16_t)((late_bits << 1) | late); on_time_run = late ? 0 : on_time_run + 1;
        g_fg_ema = 0.9 * g_fg_ema + 0.1 * (dt - want); }
    if (!g_fg_paused && __builtin_popcount(late_bits) >= 12) { g_fg_paused = 1; MLOG("frame generation: paused, the game's frames keep coming late (%.1f ms later than their step on average)", g_fg_ema); }
    else if (g_fg_paused && on_time_run >= 45) { g_fg_paused = 0; g_fg_n = 0; late_bits = 0; MLOG("frame generation: resumed (frames on time for 1.5 s)"); }
    if (g_fg_paused) return nil;
    int two = g_fg_dips && period > 0.045 && period < 0.055;
    id<MTLTexture> src = tex_mtl(t); if (!src) return nil;
    NSUInteger cx = 0, cy = 0, cw = src.width, ch = src.height;
    if (crop && crop[2] > 0 && crop[3] > 0 && crop[0] >= 0 && crop[1] >= 0 && (NSUInteger)(crop[0] + crop[2]) <= src.width && (NSUInteger)(crop[1] + crop[3]) <= src.height) {
        cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3]; }
    if (src.pixelFormat != MTLPixelFormatRGBA8Unorm) { MLOG("[RED] frame generation: the game's picture has format %lu, not RGBA8Unorm (70); off", (unsigned long)src.pixelFormat); g_fg_on = 0; return nil; }
    if (!g_fg || g_fg_ring[0].width != cw || g_fg_ring[0].height != ch) {
        fg_destroy(g_fg); g_fg = NULL; NSString *err = nil; uint64_t t0 = now_ns();
        g_fg = fg_create(g_dev, nil, (uint32_t)cw, (uint32_t)ch, 1.0f / g_fg_flow, g_fg_quality, &err);
        if (!g_fg) { MLOG("[RED] frame generation: %s; off", err.UTF8String); g_fg_on = 0; return nil; }
        if (!g_fg_q) { g_fg_q = [g_dev newCommandQueue]; g_fg_ev = [g_dev newSharedEvent]; }
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:cw height:ch mipmapped:NO];
        d.storageMode = MTLStorageModePrivate; d.usage = MTLTextureUsageShaderRead;
        for (int i = 0; i < 4; i++) { g_fg_ring[i] = [g_dev newTextureWithDescriptor:d]; g_fg_ring[i].label = @"generated frame"; res_add(g_fg_ring[i]); }
        res_commit(); g_fg_n = 0;
        MLOG("frame generation: %lux%lu pictures (the shown part of %lux%lu), set up in %.1f ms", (unsigned long)cw, (unsigned long)ch, (unsigned long)src.width, (unsigned long)src.height, (now_ns() - t0) / 1e6);
    }
    uint64_t v = g_fg_v += 2;
    [g_q4 signalEvent:g_fg_ev value:v - 1];   // after this frame's rendering on the Metal 4 queue
    id<MTLCommandBuffer> cb = [g_fg_q commandBuffer]; cb.label = @"frame generation";
    [cb encodeWaitForEvent:g_fg_ev value:v - 1];
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    [b copyFromTexture:src sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(cx, cy, 0) sourceSize:MTLSizeMake(cw, ch, 1)
             toTexture:fg_source(g_fg) destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [b endEncoding];
    id<MTLBuffer> pn = fgp_sample(cb, src, cx, cy, cw, ch);
    fg_encode_n(g_fg, cb, two ? 2 : 1);
    id<MTLTexture> out = g_fg_ring[g_fg_ri], out_b = g_fg_ring[g_fg_ri + 1];
    b = [cb blitCommandEncoder]; [b copyFromTexture:fg_output_n(g_fg, 0) toTexture:out];
    if (two) [b copyFromTexture:fg_output_n(g_fg, 1) toTexture:out_b];
    [b endEncoding];
    id<MTLBuffer> pg = pn ? fgp_sample(cb, out, 0, 0, cw, ch) : nil, pp = g_fgp_prev; g_fgp_prev = pn;
    int check = g_fg_n >= 2 && !two;   // the halfway picture made from two real frames
    [cb encodeSignalEvent:g_fg_ev value:v];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        if (c.error) { MLOG("[RED] frame generation command buffer: %s", c.error.description.UTF8String); return; }
        pthread_mutex_lock(&g_fg_mu);
        if (check && pp && pn && pg) {
            const float *P = pp.contents, *N = pn.contents, *G = pg.contents;
            g_fgp_gp += fgp_mad(G, P); g_fgp_gn += fgp_mad(G, N); g_fgp_pn += fgp_mad(P, N);
            if (++g_fgp_n == 300) {
                MLOG("frame generation check over 300 pictures (mean luma change, 0-255): generated vs previous real %.2f, vs next real %.2f, previous vs next %.2f "
                     "(in between: both about half the last; a copy of one frame: one near 0)", g_fgp_gp / 300, g_fgp_gn / 300, g_fgp_pn / 300);
                g_fgp_gp = g_fgp_gn = g_fgp_pn = 0; g_fgp_n = 0; }
        } g_fg_gpu[g_fg_gn++] = (c.GPUEndTime - c.GPUStartTime) * 1e3;
        if (g_fg_gn == 300) { double *a = g_fg_gpu; for (int i = 1; i < 300; i++) { double x = a[i]; int j = i - 1; while (j >= 0 && a[j] > x) { a[j + 1] = a[j]; j--; } a[j + 1] = x; }
            MLOG("frame generation GPU time over 300 frames: p50 %.2f ms, p95 %.2f ms, max %.2f ms", a[149], a[284], a[299]); g_fg_gn = 0; }
        pthread_mutex_unlock(&g_fg_mu); }];
    [cb commit];
    [g_q4 waitForEvent:g_fg_ev value:v];   // the present that samples the generated picture waits for it
    if (++g_fg_n < 3) return nil;          // the chain keeps two to three frames of history: nothing to show until it has them
    g_fg_ri ^= 2; if (two) *second = out_b;
    out_crop[0] = 0; out_crop[1] = 0; out_crop[2] = (int)cw; out_crop[3] = (int)ch;
    return out;
}


// DM_MFX_PROBE=1: one-off check of Apple's MetalFX frame interpolation on this device (support, and GPU time per generated frame at the game's
// 1280x896 with dummy inputs), logged as "[mfx probe]". Everything by name (NSClassFromString / KVC / objc_msgSend), so an SDK without the
// class still builds, and a device without it only logs that.
#include <objc/message.h>
static void mfx_probe(void) {
    static int done; if (done) return; done = 1;
    const char *e = getenv("DM_MFX_PROBE"); if (!e || atoi(e) <= 0) return;
    @try {
        Class D = NSClassFromString(@"MTLFXFrameInterpolatorDescriptor");
        if (!D) { MLOG("[mfx probe] MetalFX frame interpolation: not on this system"); return; }
        BOOL sup = ((BOOL (*)(id, SEL, id))objc_msgSend)(D, sel_registerName("supportsDevice:"), g_dev);
        BOOL sup4 = [D respondsToSelector:sel_registerName("supportsMetal4FX:")] ? ((BOOL (*)(id, SEL, id))objc_msgSend)(D, sel_registerName("supportsMetal4FX:"), g_dev) : NO;
        MLOG("[mfx probe] MetalFX frame interpolation: supportsDevice %d, supportsMetal4FX %d (%s)", sup, sup4, g_dev.name.UTF8String);
        if (!sup) return;
        const NSUInteger W = 1280, H = 896;
        id d = [[D alloc] init];
        [d setValue:@(MTLPixelFormatRGBA16Float) forKey:@"colorTextureFormat"]; [d setValue:@(MTLPixelFormatRGBA16Float) forKey:@"outputTextureFormat"];
        [d setValue:@(MTLPixelFormatDepth32Float) forKey:@"depthTextureFormat"]; [d setValue:@(MTLPixelFormatRG16Float) forKey:@"motionTextureFormat"];
        [d setValue:@(W) forKey:@"inputWidth"]; [d setValue:@(H) forKey:@"inputHeight"]; [d setValue:@(W) forKey:@"outputWidth"]; [d setValue:@(H) forKey:@"outputHeight"];
        uint64_t t0 = now_ns();
        id fi = ((id (*)(id, SEL, id))objc_msgSend)(d, sel_registerName("newFrameInterpolatorWithDevice:"), g_dev);
        if (!fi) { MLOG("[mfx probe] could not make a frame interpolator"); return; }
        MLOG("[mfx probe] interpolator made in %.1f ms", (now_ns() - t0) / 1e6);
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:W height:H mipmapped:NO];
        td.storageMode = MTLStorageModePrivate; td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget;
        id<MTLTexture> c0 = [g_dev newTextureWithDescriptor:td], c1 = [g_dev newTextureWithDescriptor:td], out = [g_dev newTextureWithDescriptor:td];
        td.pixelFormat = MTLPixelFormatDepth32Float; id<MTLTexture> dep = [g_dev newTextureWithDescriptor:td];
        td.pixelFormat = MTLPixelFormatRG16Float; id<MTLTexture> mot = [g_dev newTextureWithDescriptor:td];
        id<MTLCommandQueue> q = [g_dev newCommandQueue];
        double ms[12]; int n = 0;
        for (int i = 0; i < 12; i++) {
            [fi setValue:(i & 1 ? c1 : c0) forKey:@"colorTexture"]; [fi setValue:(i & 1 ? c0 : c1) forKey:@"prevColorTexture"];
            [fi setValue:dep forKey:@"depthTexture"]; [fi setValue:mot forKey:@"motionTexture"]; [fi setValue:out forKey:@"outputTexture"];
            [fi setValue:@((float)W) forKey:@"motionVectorScaleX"]; [fi setValue:@((float)H) forKey:@"motionVectorScaleY"];
            [fi setValue:@(1.0f / 30) forKey:@"deltaTime"]; [fi setValue:@(0.1f) forKey:@"nearPlane"]; [fi setValue:@(25000.0f) forKey:@"farPlane"];
            [fi setValue:@(50.0f) forKey:@"fieldOfView"]; [fi setValue:@((float)W / H) forKey:@"aspectRatio"]; [fi setValue:@(i == 0) forKey:@"shouldResetHistory"];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            ((void (*)(id, SEL, id))objc_msgSend)(fi, sel_registerName("encodeToCommandBuffer:"), cb);
            [cb commit]; [cb waitUntilCompleted];
            if (cb.error) { MLOG("[mfx probe] encode %d: %s", i, cb.error.description.UTF8String); return; }
            if (i >= 2) ms[n++] = (cb.GPUEndTime - cb.GPUStartTime) * 1e3;
        }
        double sum = 0, mx = 0; for (int i = 0; i < n; i++) { sum += ms[i]; if (ms[i] > mx) mx = ms[i]; }
        MLOG("[mfx probe] GPU per generated frame at %lux%lu: mean %.2f ms, max %.2f ms (over %d)", (unsigned long)W, (unsigned long)H, sum / n, mx, n);
    } @catch (NSException *x) { MLOG("[mfx probe] exception: %s %s", x.name.UTF8String, x.reason.UTF8String); }
}

static double g_link_it_tin;''')

# 2. present_now shows the override
once('    id<MTLTexture> src = tex_mtl(t);\n    if (!src) { RED_ONCE(8, "present texture could not get an MTLTexture"); return 0; }',
     '    id<MTLTexture> src = g_fg_override ? g_fg_override : tex_mtl(t);\n    if (!src) { RED_ONCE(8, "present texture could not get an MTLTexture"); return 0; }')

# 3. The real frame waiting for the tick after its generated picture
once('@interface NvMetalDisplayLinkDelegate : NSObject <CAMetalDisplayLinkDelegate>',
     'static pq_item g_fg_pend; static uint64_t g_fg_pend_ticket; static int g_fg_pend_on, g_fg_pend_crop[4]; static __strong id<MTLTexture> g_fg_pend_tex;   // under g_pq_mu; g_fg_pend_tex: a second generated picture to show before the real frame\n'
     '@interface NvMetalDisplayLinkDelegate : NSObject <CAMetalDisplayLinkDelegate>')

# 4. A held tick shows the waiting real frame
once('    if (g_link_hold_until > 0 && update.targetPresentationTimestamp < g_link_hold_until - tol) { g_link_held++; pthread_mutex_unlock(&g_pq_mu); return; }',
     '''    if (g_link_hold_until > 0 && update.targetPresentationTimestamp < g_link_hold_until - tol) {
        if (g_fg_pend_on) {   // frame generation: the second generated picture if there is one, then the real frame
            pq_item p = g_fg_pend; uint64_t pt = g_fg_pend_ticket; id<MTLTexture> g2 = g_fg_pend_tex; int gc[4]; memcpy(gc, g_fg_pend_crop, sizeof gc);
            if (g2) g_fg_pend_tex = nil; else g_fg_pend_on = 0;
            pthread_mutex_unlock(&g_pq_mu);
            @autoreleasepool {
                if (g2) { g_fg_override = g2; present_now(&p.t, p.period, p.frame, pt, gc, update.drawable, update.targetPresentationTimestamp); g_fg_override = nil; }
                else { present_now(&p.t, p.period, p.frame, pt, p.crop, update.drawable, update.targetPresentationTimestamp); mtl_tex_unpin(p.t.serial); }
            }
            return;
        }
        g_link_held++; pthread_mutex_unlock(&g_pq_mu); return;
    }''')

# 5. A new frame at a 30 fps step: its generated picture first
once('    @autoreleasepool { submitted = present_now(&it.t, it.period, it.frame, ticket, it.crop, update.drawable, update.targetPresentationTimestamp); mtl_tex_unpin(it.t.serial); }',
     '''    @autoreleasepool {
        id<MTLTexture> gen = nil;
        pthread_mutex_lock(&g_pq_mu); int pend = g_fg_pend_on; pthread_mutex_unlock(&g_pq_mu);
        if (pend) {   // a real frame still waiting (its tick never came): show it now, in place of a generated picture
            pthread_mutex_lock(&g_pq_mu); pq_item p = g_fg_pend; g_fg_pend_on = 0; g_fg_pend_tex = nil; pthread_mutex_unlock(&g_pq_mu); mtl_tex_unpin(p.t.serial);
        }
        mfx_probe();
        int gcrop[4] = {0, 0, 0, 0}; id<MTLTexture> gen2 = nil;
        if (it.period > 0.025) gen = fg_make(&it.t, it.period, it.crop, gcrop, &gen2);
        if (gen) {
            g_fg_override = gen;
            submitted = present_now(&it.t, it.period, it.frame, ticket, gcrop, update.drawable, update.targetPresentationTimestamp);
            g_fg_override = nil;
            if (submitted) { pthread_mutex_lock(&g_pq_mu); g_fg_pend = it; g_fg_pend_ticket = ticket; g_fg_pend_on = 1; g_fg_pend_tex = gen2; memcpy(g_fg_pend_crop, gcrop, sizeof gcrop); pthread_mutex_unlock(&g_pq_mu); }
            else mtl_tex_unpin(it.t.serial);
        } else { submitted = present_now(&it.t, it.period, it.frame, ticket, it.crop, update.drawable, update.targetPresentationTimestamp); mtl_tex_unpin(it.t.serial); }
    }''')

open(p, 'w').write(s)
print('frame generation hooked into the display-link present')
