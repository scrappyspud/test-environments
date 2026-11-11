// SPDX-License-Identifier: GPL-3.0-or-later
// Frame generation on Metal: lsfg-vk's chain, performance or quality mode (lsfg-vk-backend src/lsfgvk.cpp and src/shaderchains,
// GPL-3.0-or-later) rebuilt on Metal compute. Each Vulkan descriptor set becomes an fg_set: its pipeline, its sampled
// textures (Metal slots 0..), storage textures (16..), samplers (0..) and constant buffer (0), dispatched with the same
// threadgroup counts. A serial compute encoder with tracked textures orders the dispatches as lsfg-vk's barriers did.
#import "fg_mtl.h"
#include <stdlib.h>

#define FG_MAXS 12
#define FG_MAXW 8

typedef struct { id<MTLComputePipelineState> pso; MTLSize tg; } fg_kern;
typedef struct {
    fg_kern *k;
    __unsafe_unretained id<MTLTexture> s[FG_MAXS]; int ns;
    __unsafe_unretained id<MTLTexture> w[FG_MAXW]; int nw;
    __unsafe_unretained id<MTLSamplerState> sm[2]; int nsm;
    __unsafe_unretained id<MTLBuffer> b;
    MTLSize groups;
} fg_set;
typedef struct { uint32_t w, h; } fg_ext2;

typedef struct {   // lsfg-vk's ConstantBuffer
    uint32_t inputOffset[2], firstIter, firstIterS, advancedColorKind, hdrSupport;
    float resolutionInvScale, timestamp, uiThreshold;
    uint32_t pad[3];
} fg_cb;

enum { FGK_MIP, FGK_GEN, FGK_A0, FGK_A1, FGK_A2, FGK_A3, FGK_B0, FGK_B1, FGK_B2, FGK_B3, FGK_B4,
       FGK_G0, FGK_G1, FGK_G2, FGK_G3, FGK_G4, FGK_D0, FGK_D1, FGK_D2, FGK_D3, FGK_D4, FGK_D5, FGK_D6, FGK_D7, FGK_D8, FGK_D9, FGK_N };
static const char *fg_kname[FGK_N] = { "mipmaps", "generate", "alpha0", "alpha1", "alpha2", "alpha3", "beta0", "beta1", "beta2", "beta3", "beta4",
    "gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "delta0", "delta1", "delta2", "delta3", "delta4", "delta5", "delta6", "delta7", "delta8", "delta9" };

struct fg_ctx {
    id<MTLDevice> dev;
    fg_kern k[FGK_N];
    NSMutableArray *keep;            // every texture, buffer and sampler the sets point at
    id<MTLSamplerState> bnb, bnw, eab;
    id<MTLBuffer> cb0, cb1;          // ctx.constantBuffer, ctx.constantBuffers[0]
    id<MTLBuffer> cbt[2];            // two pictures per pair: timestamps 1/3 and 2/3
    id<MTLTexture> src[2], out, out2, black;
    int m;                           // lsfg-vk's multiplier: 1 performance, 2 quality
    fg_ext2 sext, fext;
    fg_set *sets; int nsets, capsets;
    // the chain, as indices into sets
    int mip[2];
    int a0[7][3];
    int a1n[7], a1[7][3]; id<MTLTexture> a1img[7][3][4];
    int b0[3]; int b1[4];
    int g0[7][3], g1[7][4], d0s0[3][3], d0s1[3][3], d1[3][8];
    int gen[2];
    uint64_t fidx;
};

static fg_ext2 fg_shift(fg_ext2 e, uint32_t i) { return (fg_ext2){ e.w >> i, e.h >> i }; }
static fg_ext2 fg_ashift(fg_ext2 e, uint32_t a, uint32_t i) { return (fg_ext2){ (e.w + a) >> i, (e.h + a) >> i }; }

static id<MTLTexture> fg_img(fg_ctx *c, fg_ext2 e, MTLPixelFormat f) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:e.w ? e.w : 1 height:e.h ? e.h : 1 mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [c->dev newTextureWithDescriptor:d]; [c->keep addObject:t]; return t;
}
#define FG_RGBA8 MTLPixelFormatRGBA8Unorm
#define FG_R8 MTLPixelFormatR8Unorm
#define FG_RGBA16F MTLPixelFormatRGBA16Float

static int fg_set_new(fg_ctx *c, int k, fg_ext2 groups) {
    if (c->nsets == c->capsets) { c->capsets = c->capsets ? 2 * c->capsets : 256; c->sets = realloc(c->sets, c->capsets * sizeof(fg_set)); }
    fg_set *s = &c->sets[c->nsets]; memset(s, 0, sizeof *s); s->k = &c->k[k]; s->groups = MTLSizeMake(groups.w, groups.h, 1);
    return c->nsets++;
}
static void fg_S(fg_ctx *c, int i, id<MTLTexture> t) { fg_set *s = &c->sets[i]; if (s->ns < FG_MAXS) s->s[s->ns++] = t; }
static void fg_W(fg_ctx *c, int i, id<MTLTexture> t) { fg_set *s = &c->sets[i]; if (s->nw < FG_MAXW) s->w[s->nw++] = t; }
static void fg_SM(fg_ctx *c, int i, id<MTLSamplerState> t) { fg_set *s = &c->sets[i]; if (s->nsm < 2) s->sm[s->nsm++] = t; }
static void fg_B(fg_ctx *c, int i, id<MTLBuffer> b) { c->sets[i].b = b; }

static id<MTLSamplerState> fg_sampler(fg_ctx *c, MTLSamplerAddressMode mode, BOOL white) {
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.minFilter = d.magFilter = MTLSamplerMinMagFilterLinear; d.mipFilter = MTLSamplerMipFilterLinear;
    d.sAddressMode = d.tAddressMode = d.rAddressMode = mode;
    d.borderColor = white ? MTLSamplerBorderColorOpaqueWhite : MTLSamplerBorderColorTransparentBlack;
    id<MTLSamplerState> s = [c->dev newSamplerStateWithDescriptor:d]; [c->keep addObject:s]; return s;
}
static id<MTLBuffer> fg_cbuf(fg_ctx *c, float flow, float ts) {
    fg_cb v = {0}; v.resolutionInvScale = flow; v.timestamp = ts; v.uiThreshold = 0.5f;
    id<MTLBuffer> b = [c->dev newBufferWithBytes:&v length:sizeof v options:MTLResourceStorageModeShared]; [c->keep addObject:b]; return b;
}

#ifdef FG_EMBEDDED
#include "fg_blob.h"   // fg_blobs[]: role, metallib bytes, threadgroup size (written by gen_blob.py at build time)
#endif
static BOOL fg_load_kernels(fg_ctx *c, NSString *dir, NSString **err) {
    NSDictionary *man = nil;
    if (dir) {
        NSData *mj = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
        man = mj ? [NSJSONSerialization JSONObjectWithData:mj options:0 error:nil] : nil;
        if (!man) { if (err) *err = @"manifest.json missing"; return NO; }
    }
    for (int i = 0; i < FGK_N; i++) {
        NSString *role = c->m == 2 && i >= FGK_A0 ? [@"q_" stringByAppendingString:@(fg_kname[i])] : @(fg_kname[i]); NSError *e = nil; id<MTLLibrary> lib = nil; unsigned tg[3] = {0, 0, 0};
        if (dir) {
            lib = [c->dev newLibraryWithURL:[NSURL fileURLWithPath:[dir stringByAppendingPathComponent:[role stringByAppendingString:@".metallib"]]] error:&e];
            NSArray *a = man[role][@"tg"]; for (int k = 0; k < 3; k++) tg[k] = [a[k] unsignedIntValue];
        } else {
#ifdef FG_EMBEDDED
            for (size_t b = 0; b < sizeof fg_blobs / sizeof *fg_blobs; b++) if (!strcmp(fg_blobs[b].role, role.UTF8String)) {
                dispatch_data_t dd = dispatch_data_create(fg_blobs[b].data, fg_blobs[b].len, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
                lib = [c->dev newLibraryWithData:dd error:&e]; memcpy(tg, fg_blobs[b].tg, sizeof tg);
            }
#endif
        }
        id<MTLFunction> f = [lib newFunctionWithName:[@"k_" stringByAppendingString:role]];
        if (!f) { if (err) *err = [NSString stringWithFormat:@"%@: %@", role, e.localizedDescription ?: @"no function"]; return NO; }
        c->k[i].pso = [c->dev newComputePipelineStateWithFunction:f error:&e];
        if (!c->k[i].pso) { if (err) *err = [NSString stringWithFormat:@"%@ pipeline: %@", role, e.localizedDescription]; return NO; }
        c->k[i].tg = MTLSizeMake(tg[0], tg[1], tg[2]);
        if (!tg[0] || c->k[i].tg.width * c->k[i].tg.height > c->k[i].pso.maxTotalThreadsPerThreadgroup) {
            if (err) *err = [NSString stringWithFormat:@"%@: threadgroup %ux%u over the pipeline's %lu", role, tg[0], tg[1], (unsigned long)c->k[i].pso.maxTotalThreadsPerThreadgroup];
            return NO;
        }
    }
    return YES;
}

fg_ctx *fg_create(id<MTLDevice> dev, NSString *dir, uint32_t width, uint32_t height, float flow, int quality, NSString **err) {
    fg_ctx *c = calloc(1, sizeof *c);
    c->dev = dev; c->keep = [NSMutableArray array]; c->m = quality ? 2 : 1;
    const int m = c->m;
    if (!fg_load_kernels(c, dir, err)) { fg_destroy(c); return NULL; }
    c->sext = (fg_ext2){ width, height };
    c->fext = (fg_ext2){ (uint32_t)((float)width / flow), (uint32_t)((float)height / flow) };
    c->bnb = fg_sampler(c, MTLSamplerAddressModeClampToBorderColor, NO);
    c->bnw = fg_sampler(c, MTLSamplerAddressModeClampToBorderColor, YES);
    c->eab = fg_sampler(c, MTLSamplerAddressModeClampToEdge, NO);
    c->cb0 = fg_cbuf(c, flow, 0.5f); c->cb1 = fg_cbuf(c, flow, 0.5f);   // getDefaultConstantBuffer(0, 1) and (0, count = 1)
    c->cbt[0] = fg_cbuf(c, flow, 1.0f / 3.0f); c->cbt[1] = fg_cbuf(c, flow, 2.0f / 3.0f);   // (0, 2) and (1, 2): timestamp (i + 1) / (count + 1)
    c->src[0] = fg_img(c, c->sext, FG_RGBA8); c->src[1] = fg_img(c, c->sext, FG_RGBA8);
    c->out = fg_img(c, c->sext, FG_RGBA8); c->out2 = fg_img(c, c->sext, FG_RGBA8);
    {   // the black image: lsfg-vk never writes it; zero it so it reads as transparent black
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:FG_RGBA8 width:4 height:4 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModeShared;
        c->black = [dev newTextureWithDescriptor:d]; uint8_t z[64] = {0};
        [c->black replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:z bytesPerRow:16]; [c->keep addObject:c->black];
    }

    // Mipmaps
    id<MTLTexture> mips[7];
    for (int i = 0; i < 7; i++) mips[i] = fg_img(c, fg_shift(c->fext, i), FG_R8);
    for (int s = 0; s < 2; s++) {
        int x = c->mip[s] = fg_set_new(c, FGK_MIP, fg_ashift(c->fext, 63, 6));
        fg_S(c, x, c->src[s]); for (int i = 0; i < 7; i++) fg_W(c, x, mips[i]); fg_SM(c, x, c->bnb); fg_B(c, x, c->cb0);
    }
    // Alpha0 (m temp pairs, 2m outputs) and Alpha1 per mip level
    id<MTLTexture> a0img[7][4];
    for (int l = 0; l < 7; l++) {
        fg_ext2 half = fg_ashift(mips[l].width ? (fg_ext2){ (uint32_t)mips[l].width, (uint32_t)mips[l].height } : c->fext, 1, 1), quarter = fg_ashift(half, 1, 1);
        id<MTLTexture> t0[2], t1[2];
        for (int k = 0; k < m; k++) { t0[k] = fg_img(c, half, FG_RGBA8); t1[k] = fg_img(c, half, FG_RGBA8); }
        for (int k = 0; k < 2 * m; k++) a0img[l][k] = fg_img(c, quarter, FG_RGBA8);
        int x;
        x = c->a0[l][0] = fg_set_new(c, FGK_A0, fg_ashift(half, 7, 3)); fg_S(c, x, mips[l]); for (int k = 0; k < m; k++) fg_W(c, x, t0[k]); fg_SM(c, x, c->bnb);
        x = c->a0[l][1] = fg_set_new(c, FGK_A1, fg_ashift(half, 7, 3)); for (int k = 0; k < m; k++) { fg_S(c, x, t0[k]); fg_W(c, x, t1[k]); } fg_SM(c, x, c->bnb);
        x = c->a0[l][2] = fg_set_new(c, FGK_A2, fg_ashift(quarter, 7, 3)); for (int k = 0; k < m; k++) fg_S(c, x, t1[k]); for (int k = 0; k < 2 * m; k++) fg_W(c, x, a0img[l][k]); fg_SM(c, x, c->bnb);
        int T = c->a1n[l] = l == 0 ? 3 : 2;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < 2 * m; k++) c->a1img[l][t][k] = fg_img(c, quarter, FG_RGBA8);
            x = c->a1[l][t] = fg_set_new(c, FGK_A3, fg_ashift(quarter, 7, 3));
            for (int k = 0; k < 2 * m; k++) fg_S(c, x, a0img[l][k]);
            for (int k = 0; k < 2 * m; k++) fg_W(c, x, c->a1img[l][t][k]);
            fg_SM(c, x, c->bnb);
        }
    }
    // Beta0 (over alpha1 of level 0, 3 temporal slots) and Beta1
    fg_ext2 be = { (uint32_t)c->a1img[0][0][0].width, (uint32_t)c->a1img[0][0][0].height };
    id<MTLTexture> b0img[2] = { fg_img(c, be, FG_RGBA8), fg_img(c, be, FG_RGBA8) };
    for (int i = 0; i < 3; i++) {
        int x = c->b0[i] = fg_set_new(c, FGK_B0, fg_ashift(be, 7, 3));
        int o[3] = { (i + 1) % 3, (i + 2) % 3, i % 3 };
        for (int q = 0; q < 3; q++) for (int k = 0; k < 2 * m; k++) fg_S(c, x, c->a1img[0][o[q]][k]);
        fg_W(c, x, b0img[0]); fg_W(c, x, b0img[1]); fg_SM(c, x, c->bnw);
    }
    id<MTLTexture> bt0[2] = { fg_img(c, be, FG_RGBA8), fg_img(c, be, FG_RGBA8) }, bt1[2] = { fg_img(c, be, FG_RGBA8), fg_img(c, be, FG_RGBA8) }, b1img[6];
    for (int i = 0; i < 6; i++) b1img[i] = fg_img(c, fg_shift(be, i), FG_R8);
    {
        int x;
        x = c->b1[0] = fg_set_new(c, FGK_B1, fg_ashift(be, 7, 3)); fg_S(c, x, b0img[0]); fg_S(c, x, b0img[1]); fg_W(c, x, bt0[0]); fg_W(c, x, bt0[1]); fg_SM(c, x, c->bnb);
        x = c->b1[1] = fg_set_new(c, FGK_B2, fg_ashift(be, 7, 3)); fg_S(c, x, bt0[0]); fg_S(c, x, bt0[1]); fg_W(c, x, bt1[0]); fg_W(c, x, bt1[1]); fg_SM(c, x, c->bnb);
        x = c->b1[2] = fg_set_new(c, FGK_B3, fg_ashift(be, 7, 3)); fg_S(c, x, bt1[0]); fg_S(c, x, bt1[1]); fg_W(c, x, bt0[0]); fg_W(c, x, bt0[1]); fg_SM(c, x, c->bnb);
        x = c->b1[3] = fg_set_new(c, FGK_B4, fg_ashift(be, 31, 5)); fg_S(c, x, bt0[0]); fg_S(c, x, bt0[1]); for (int i = 0; i < 6; i++) fg_W(c, x, b1img[i]); fg_SM(c, x, c->bnb); fg_B(c, x, c->cb0);
    }
    // The generation pass (count = 1): gamma over all 7 levels (coarse to fine), delta over the 3 finest
    id<MTLTexture> g1img[7], d1img0[3], d1img1[3];
    for (int j = 0; j < 7; j++) {
        int l = 6 - j, T = c->a1n[l];
        fg_ext2 e = { (uint32_t)c->a1img[l][0][0].width, (uint32_t)c->a1img[l][0][0].height };
        id<MTLTexture> g0img[3] = { fg_img(c, e, FG_RGBA8), fg_img(c, e, FG_RGBA8), fg_img(c, e, FG_RGBA8) };
        id<MTLTexture> gprev = j == 0 ? c->black : g1img[j - 1];
        for (int i = 0; i < T; i++) {   // Gamma0
            int x = c->g0[j][i] = fg_set_new(c, FGK_G0, fg_ashift(e, 7, 3)), p = (i + T - 1) % T, q = i % T;
            for (int k = 0; k < 2 * m; k++) fg_S(c, x, c->a1img[l][p][k]); for (int k = 0; k < 2 * m; k++) fg_S(c, x, c->a1img[l][q][k]); fg_S(c, x, gprev);
            for (int k = 0; k < 3; k++) fg_W(c, x, g0img[k]);
            fg_SM(c, x, c->bnw); fg_SM(c, x, c->eab); fg_B(c, x, c->cb1);
        }
        {   // Gamma1
            id<MTLTexture> t0[4], t1[4]; const int n = 2 * m;
            for (int k = 0; k < n; k++) { t0[k] = fg_img(c, e, FG_RGBA8); t1[k] = fg_img(c, e, FG_RGBA8); }
            g1img[j] = fg_img(c, e, FG_RGBA16F);
            id<MTLTexture> bet = b1img[j == 0 ? 5 : 6 - j];
            int x;
            x = c->g1[j][0] = fg_set_new(c, FGK_G1, fg_ashift(e, 7, 3)); for (int k = 0; k < 3; k++) fg_S(c, x, g0img[k]); for (int k = 0; k < n; k++) fg_W(c, x, t0[k]); fg_SM(c, x, c->bnb);
            x = c->g1[j][1] = fg_set_new(c, FGK_G2, fg_ashift(e, 7, 3)); for (int k = 0; k < n; k++) { fg_S(c, x, t0[k]); fg_W(c, x, t1[k]); } fg_SM(c, x, c->bnb);
            x = c->g1[j][2] = fg_set_new(c, FGK_G3, fg_ashift(e, 7, 3)); for (int k = 0; k < n; k++) { fg_S(c, x, t1[k]); fg_W(c, x, t0[k]); } fg_SM(c, x, c->bnb);
            x = c->g1[j][3] = fg_set_new(c, FGK_G4, fg_ashift(e, 7, 3)); for (int k = 0; k < n; k++) fg_S(c, x, t0[k]); fg_S(c, x, gprev); fg_S(c, x, bet); fg_W(c, x, g1img[j]);
            fg_SM(c, x, c->bnb); fg_SM(c, x, c->eab); fg_B(c, x, c->cb1);
        }
        if (j < 4) continue;
        int dj = j - 4;
        id<MTLTexture> dadd0 = dj == 0 ? c->black : d1img0[dj - 1], dadd1 = g1img[j - 1];
        id<MTLTexture> d0i0[3] = { fg_img(c, e, FG_RGBA8), fg_img(c, e, FG_RGBA8), fg_img(c, e, FG_RGBA8) }, d0i1[2] = { fg_img(c, e, FG_RGBA8), m == 2 ? fg_img(c, e, FG_RGBA8) : nil };
        for (int i = 0; i < T; i++) {   // Delta0
            int p = (i + T - 1) % T, q = i % T, x;
            x = c->d0s0[dj][i] = fg_set_new(c, FGK_D0, fg_ashift(e, 7, 3));
            for (int k = 0; k < 2 * m; k++) fg_S(c, x, c->a1img[l][p][k]); for (int k = 0; k < 2 * m; k++) fg_S(c, x, c->a1img[l][q][k]); fg_S(c, x, dadd0);
            for (int k = 0; k < 3; k++) fg_W(c, x, d0i0[k]);
            fg_SM(c, x, c->bnw); fg_SM(c, x, c->eab); fg_B(c, x, c->cb1);
            x = c->d0s1[dj][i] = fg_set_new(c, FGK_D5, fg_ashift(e, 7, 3));
            for (int k = 0; k < 2 * m; k++) fg_S(c, x, c->a1img[l][p][k]); for (int k = 0; k < 2 * m; k++) fg_S(c, x, c->a1img[l][q][k]); fg_S(c, x, dadd1); fg_S(c, x, dadd0);
            for (int k = 0; k < m; k++) fg_W(c, x, d0i1[k]);
            fg_SM(c, x, c->bnw); fg_SM(c, x, c->eab); fg_B(c, x, c->cb1);
        }
        {   // Delta1
            id<MTLTexture> a0 = dj == 0 ? c->black : d1img0[dj - 1], a1 = b1img[6 - j], a2 = dj == 0 ? c->black : d1img1[dj - 1];
            id<MTLTexture> t0[4], t1[4]; const int n = 2 * m;
            for (int k = 0; k < n; k++) { t0[k] = fg_img(c, e, FG_RGBA8); t1[k] = fg_img(c, e, FG_RGBA8); }
            d1img0[dj] = fg_img(c, e, FG_RGBA16F); d1img1[dj] = fg_img(c, e, FG_RGBA16F);
            int x, *d = c->d1[dj];
            x = d[0] = fg_set_new(c, FGK_D1, fg_ashift(e, 7, 3)); for (int k = 0; k < 3; k++) fg_S(c, x, d0i0[k]); for (int k = 0; k < n; k++) fg_W(c, x, t0[k]); fg_SM(c, x, c->bnb);
            x = d[1] = fg_set_new(c, FGK_D2, fg_ashift(e, 7, 3)); for (int k = 0; k < n; k++) { fg_S(c, x, t0[k]); fg_W(c, x, t1[k]); } fg_SM(c, x, c->bnb);
            x = d[2] = fg_set_new(c, FGK_D3, fg_ashift(e, 7, 3)); for (int k = 0; k < n; k++) { fg_S(c, x, t1[k]); fg_W(c, x, t0[k]); } fg_SM(c, x, c->bnb);
            x = d[3] = fg_set_new(c, FGK_D4, fg_ashift(e, 7, 3)); for (int k = 0; k < n; k++) fg_S(c, x, t0[k]); fg_S(c, x, a0); fg_S(c, x, a1); fg_W(c, x, d1img0[dj]);
            fg_SM(c, x, c->bnb); fg_SM(c, x, c->eab); fg_B(c, x, c->cb1);
            x = d[4] = fg_set_new(c, FGK_D6, fg_ashift(e, 7, 3)); for (int k = 0; k < m; k++) { fg_S(c, x, d0i1[k]); fg_W(c, x, t0[k]); } fg_SM(c, x, c->bnb);
            x = d[5] = fg_set_new(c, FGK_D7, fg_ashift(e, 7, 3)); for (int k = 0; k < m; k++) { fg_S(c, x, t0[k]); fg_W(c, x, t1[k]); } fg_SM(c, x, c->bnb);
            x = d[6] = fg_set_new(c, FGK_D8, fg_ashift(e, 7, 3)); for (int k = 0; k < m; k++) { fg_S(c, x, t1[k]); fg_W(c, x, t0[k]); } fg_SM(c, x, c->bnb);
            x = d[7] = fg_set_new(c, FGK_D9, fg_ashift(e, 7, 3)); for (int k = 0; k < m; k++) fg_S(c, x, t0[k]); fg_S(c, x, a2); fg_W(c, x, d1img1[dj]);
            fg_SM(c, x, c->bnb); fg_SM(c, x, c->eab); fg_B(c, x, c->cb1);
        }
    }
    for (int s = 0; s < 2; s++) {   // Generate: set 0 reads (second, first), set 1 (first, second)
        int x = c->gen[s] = fg_set_new(c, FGK_GEN, fg_ashift(c->sext, 15, 4));
        fg_S(c, x, c->src[s ? 0 : 1]); fg_S(c, x, c->src[s ? 1 : 0]); fg_S(c, x, g1img[6]); fg_S(c, x, d1img0[2]); fg_S(c, x, d1img1[2]);
        fg_W(c, x, c->out); fg_SM(c, x, c->bnb); fg_SM(c, x, c->eab); fg_B(c, x, c->cb1);
    }
    return c;
}

id<MTLTexture> fg_source(fg_ctx *c) { return c->src[c->fidx % 2]; }
id<MTLTexture> fg_output(fg_ctx *c) { return c->out; }
id<MTLTexture> fg_output_n(fg_ctx *c, int i) { return i ? c->out2 : c->out; }

// pb, po: this pass's constant buffer and output in place of the set's (ctx.constantBuffers[0] and the first output)
static void fg_run2(fg_ctx *c, id<MTLComputeCommandEncoder> e, const fg_set *s, id<MTLBuffer> pb, id<MTLTexture> po) {
    [e setComputePipelineState:s->k->pso];
    for (int i = 0; i < s->ns; i++) [e setTexture:s->s[i] atIndex:i];
    for (int i = 0; i < s->nw; i++) [e setTexture:po && s->w[i] == c->out ? po : s->w[i] atIndex:16 + i];
    for (int i = 0; i < s->nsm; i++) [e setSamplerState:s->sm[i] atIndex:i];
    if (s->b) [e setBuffer:pb && s->b == c->cb1 ? pb : s->b offset:0 atIndex:0];
    [e dispatchThreadgroups:s->groups threadsPerThreadgroup:s->k->tg];
}
#define fg_run(e, s) fg_run2(c, e, s, pb, po)

void fg_encode(fg_ctx *c, id<MTLCommandBuffer> cb) { fg_encode_n(c, cb, 1); }
// n = 1: the picture halfway, in fg_output. n = 2: the pictures a third and two thirds of the way, in fg_output_n(0) and (1)
void fg_encode_n(fg_ctx *c, id<MTLCommandBuffer> cb, int n) {
    id<MTLBuffer> pb = nil; id<MTLTexture> po = nil;
    uint64_t f = c->fidx;
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    e.label = @"fg";
    fg_run(e, &c->sets[c->mip[f % 2]]);
    for (int i = 0; i < 7; i++) {
        int l = 6 - i;
        for (int k = 0; k < 3; k++) fg_run(e, &c->sets[c->a0[l][k]]);
        fg_run(e, &c->sets[c->a1[l][f % c->a1n[l]]]);
    }
    fg_run(e, &c->sets[c->b0[f % 3]]);
    for (int k = 0; k < 4; k++) fg_run(e, &c->sets[c->b1[k]]);
    for (int pass = 0; pass < (n == 2 ? 2 : 1); pass++) {
    if (n == 2) { pb = c->cbt[pass]; po = pass ? c->out2 : c->out; }
    for (int j = 0; j < 7; j++) {
        int T = c->a1n[6 - j];
        fg_run(e, &c->sets[c->g0[j][f % T]]);
        for (int k = 0; k < 4; k++) fg_run(e, &c->sets[c->g1[j][k]]);
        if (j < 4) continue;
        fg_run(e, &c->sets[c->d0s0[j - 4][f % T]]);
        fg_run(e, &c->sets[c->d0s1[j - 4][f % T]]);
        for (int k = 0; k < 8; k++) fg_run(e, &c->sets[c->d1[j - 4][k]]);
    }
    fg_run(e, &c->sets[c->gen[f % 2]]);
    }
    [e endEncoding];
    c->fidx++;
}

void fg_destroy(fg_ctx *c) {
    if (!c) return;
    for (int i = 0; i < FGK_N; i++) c->k[i].pso = nil;
    c->keep = nil; c->dev = nil; c->bnb = c->bnw = c->eab = nil; c->cb0 = c->cb1 = nil;
    c->src[0] = c->src[1] = c->out = c->out2 = c->black = nil; c->cbt[0] = c->cbt[1] = nil;
    free(c->sets); free(c);
}
