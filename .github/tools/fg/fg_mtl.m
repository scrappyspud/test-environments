// SPDX-License-Identifier: GPL-3.0-or-later
// Frame generation on Metal: lsfg-vk's performance-mode chain (lsfg-vk-backend src/lsfgvk.cpp and src/shaderchains,
// GPL-3.0-or-later) rebuilt on Metal compute. Each Vulkan descriptor set becomes an fg_set: its pipeline, its sampled
// textures (Metal slots 0..), storage textures (16..), samplers (0..) and constant buffer (0), dispatched with the same
// threadgroup counts. A serial compute encoder with tracked textures orders the dispatches as lsfg-vk's barriers did.
#import "fg_mtl.h"
#include <stdlib.h>

#define MAXS 12
#define MAXW 8

typedef struct { id<MTLComputePipelineState> pso; MTLSize tg; } fg_kern;
typedef struct {
    fg_kern *k;
    __unsafe_unretained id<MTLTexture> s[MAXS]; int ns;
    __unsafe_unretained id<MTLTexture> w[MAXW]; int nw;
    __unsafe_unretained id<MTLSamplerState> sm[2]; int nsm;
    __unsafe_unretained id<MTLBuffer> b;
    MTLSize groups;
} fg_set;
typedef struct { uint32_t w, h; } ext2;

typedef struct {   // lsfg-vk's ConstantBuffer
    uint32_t inputOffset[2], firstIter, firstIterS, advancedColorKind, hdrSupport;
    float resolutionInvScale, timestamp, uiThreshold;
    uint32_t pad[3];
} fg_cb;

enum { K_MIP, K_GEN, K_A0, K_A1, K_A2, K_A3, K_B0, K_B1, K_B2, K_B3, K_B4,
       K_G0, K_G1, K_G2, K_G3, K_G4, K_D0, K_D1, K_D2, K_D3, K_D4, K_D5, K_D6, K_D7, K_D8, K_D9, K_N };
static const char *kname[K_N] = { "mipmaps", "generate", "alpha0", "alpha1", "alpha2", "alpha3", "beta0", "beta1", "beta2", "beta3", "beta4",
    "gamma0", "gamma1", "gamma2", "gamma3", "gamma4", "delta0", "delta1", "delta2", "delta3", "delta4", "delta5", "delta6", "delta7", "delta8", "delta9" };

struct fg_ctx {
    id<MTLDevice> dev;
    fg_kern k[K_N];
    NSMutableArray *keep;            // every texture, buffer and sampler the sets point at
    id<MTLSamplerState> bnb, bnw, eab;
    id<MTLBuffer> cb0, cb1;          // ctx.constantBuffer, ctx.constantBuffers[0]
    id<MTLTexture> src[2], out, black;
    ext2 sext, fext;
    fg_set *sets; int nsets, capsets;
    // the chain, as indices into sets
    int mip[2];
    int a0[7][3];
    int a1n[7], a1[7][3]; id<MTLTexture> a1img[7][3][2];
    int b0[3]; int b1[4];
    int g0[7][3], g1[7][4], d0s0[3][3], d0s1[3][3], d1[3][8];
    int gen[2];
    uint64_t fidx;
};

static ext2 shift(ext2 e, uint32_t i) { return (ext2){ e.w >> i, e.h >> i }; }
static ext2 ashift(ext2 e, uint32_t a, uint32_t i) { return (ext2){ (e.w + a) >> i, (e.h + a) >> i }; }

static id<MTLTexture> img(fg_ctx *c, ext2 e, MTLPixelFormat f) {
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:e.w ? e.w : 1 height:e.h ? e.h : 1 mipmapped:NO];
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> t = [c->dev newTextureWithDescriptor:d]; [c->keep addObject:t]; return t;
}
#define RGBA8 MTLPixelFormatRGBA8Unorm
#define R8 MTLPixelFormatR8Unorm
#define RGBA16F MTLPixelFormatRGBA16Float

static int set_new(fg_ctx *c, int k, ext2 groups) {
    if (c->nsets == c->capsets) { c->capsets = c->capsets ? 2 * c->capsets : 256; c->sets = realloc(c->sets, c->capsets * sizeof(fg_set)); }
    fg_set *s = &c->sets[c->nsets]; memset(s, 0, sizeof *s); s->k = &c->k[k]; s->groups = MTLSizeMake(groups.w, groups.h, 1);
    return c->nsets++;
}
static void S(fg_ctx *c, int i, id<MTLTexture> t) { fg_set *s = &c->sets[i]; if (s->ns < MAXS) s->s[s->ns++] = t; }
static void W(fg_ctx *c, int i, id<MTLTexture> t) { fg_set *s = &c->sets[i]; if (s->nw < MAXW) s->w[s->nw++] = t; }
static void SM(fg_ctx *c, int i, id<MTLSamplerState> t) { fg_set *s = &c->sets[i]; if (s->nsm < 2) s->sm[s->nsm++] = t; }
static void B(fg_ctx *c, int i, id<MTLBuffer> b) { c->sets[i].b = b; }

static id<MTLSamplerState> sampler(fg_ctx *c, MTLSamplerAddressMode mode, BOOL white) {
    MTLSamplerDescriptor *d = [MTLSamplerDescriptor new];
    d.minFilter = d.magFilter = MTLSamplerMinMagFilterLinear; d.mipFilter = MTLSamplerMipFilterLinear;
    d.sAddressMode = d.tAddressMode = d.rAddressMode = mode;
    d.borderColor = white ? MTLSamplerBorderColorOpaqueWhite : MTLSamplerBorderColorTransparentBlack;
    id<MTLSamplerState> s = [c->dev newSamplerStateWithDescriptor:d]; [c->keep addObject:s]; return s;
}
static id<MTLBuffer> cbuf(fg_ctx *c, float flow, float ts) {
    fg_cb v = {0}; v.resolutionInvScale = flow; v.timestamp = ts; v.uiThreshold = 0.5f;
    id<MTLBuffer> b = [c->dev newBufferWithBytes:&v length:sizeof v options:MTLResourceStorageModeShared]; [c->keep addObject:b]; return b;
}

static BOOL load_kernels(fg_ctx *c, NSString *dir, NSString **err) {
    NSData *mj = [NSData dataWithContentsOfFile:[dir stringByAppendingPathComponent:@"manifest.json"]];
    NSDictionary *man = mj ? [NSJSONSerialization JSONObjectWithData:mj options:0 error:nil] : nil;
    if (!man) { if (err) *err = @"manifest.json missing"; return NO; }
    for (int i = 0; i < K_N; i++) {
        NSString *role = @(kname[i]); NSError *e = nil;
        id<MTLLibrary> lib = [c->dev newLibraryWithURL:[NSURL fileURLWithPath:[dir stringByAppendingPathComponent:[role stringByAppendingString:@".metallib"]]] error:&e];
        id<MTLFunction> f = [lib newFunctionWithName:[@"k_" stringByAppendingString:role]];
        if (!f) { if (err) *err = [NSString stringWithFormat:@"%@: %@", role, e.localizedDescription ?: @"no function"]; return NO; }
        c->k[i].pso = [c->dev newComputePipelineStateWithFunction:f error:&e];
        if (!c->k[i].pso) { if (err) *err = [NSString stringWithFormat:@"%@ pipeline: %@", role, e.localizedDescription]; return NO; }
        NSArray *tg = man[role][@"tg"];
        c->k[i].tg = MTLSizeMake([tg[0] unsignedIntegerValue], [tg[1] unsignedIntegerValue], [tg[2] unsignedIntegerValue]);
        if (c->k[i].tg.width * c->k[i].tg.height > c->k[i].pso.maxTotalThreadsPerThreadgroup) {
            if (err) *err = [NSString stringWithFormat:@"%@: threadgroup %lux%lu over the pipeline's %lu", role, (unsigned long)c->k[i].tg.width, (unsigned long)c->k[i].tg.height, (unsigned long)c->k[i].pso.maxTotalThreadsPerThreadgroup];
            return NO;
        }
    }
    return YES;
}

fg_ctx *fg_create(id<MTLDevice> dev, NSString *dir, uint32_t width, uint32_t height, float flow, NSString **err) {
    fg_ctx *c = calloc(1, sizeof *c);
    c->dev = dev; c->keep = [NSMutableArray array];
    if (!load_kernels(c, dir, err)) { fg_destroy(c); return NULL; }
    c->sext = (ext2){ width, height };
    c->fext = (ext2){ (uint32_t)((float)width / flow), (uint32_t)((float)height / flow) };
    c->bnb = sampler(c, MTLSamplerAddressModeClampToBorderColor, NO);
    c->bnw = sampler(c, MTLSamplerAddressModeClampToBorderColor, YES);
    c->eab = sampler(c, MTLSamplerAddressModeClampToEdge, NO);
    c->cb0 = cbuf(c, flow, 0.5f); c->cb1 = cbuf(c, flow, 0.5f);   // getDefaultConstantBuffer(0, 1) and (0, count = 1)
    c->src[0] = img(c, c->sext, RGBA8); c->src[1] = img(c, c->sext, RGBA8);
    c->out = img(c, c->sext, RGBA8);
    {   // the black image: lsfg-vk never writes it; zero it so it reads as transparent black
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:RGBA8 width:4 height:4 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModeShared;
        c->black = [dev newTextureWithDescriptor:d]; uint8_t z[64] = {0};
        [c->black replaceRegion:MTLRegionMake2D(0, 0, 4, 4) mipmapLevel:0 withBytes:z bytesPerRow:16]; [c->keep addObject:c->black];
    }

    // Mipmaps
    id<MTLTexture> mips[7];
    for (int i = 0; i < 7; i++) mips[i] = img(c, shift(c->fext, i), R8);
    for (int s = 0; s < 2; s++) {
        int x = c->mip[s] = set_new(c, K_MIP, ashift(c->fext, 63, 6));
        S(c, x, c->src[s]); for (int i = 0; i < 7; i++) W(c, x, mips[i]); SM(c, x, c->bnb); B(c, x, c->cb0);
    }
    // Alpha0 (perf: one temp pair, two outputs) and Alpha1 per mip level
    id<MTLTexture> a0img[7][2];
    for (int l = 0; l < 7; l++) {
        ext2 half = ashift(mips[l].width ? (ext2){ (uint32_t)mips[l].width, (uint32_t)mips[l].height } : c->fext, 1, 1), quarter = ashift(half, 1, 1);
        id<MTLTexture> t0 = img(c, half, RGBA8), t1 = img(c, half, RGBA8);
        a0img[l][0] = img(c, quarter, RGBA8); a0img[l][1] = img(c, quarter, RGBA8);
        int x;
        x = c->a0[l][0] = set_new(c, K_A0, ashift(half, 7, 3)); S(c, x, mips[l]); W(c, x, t0); SM(c, x, c->bnb);
        x = c->a0[l][1] = set_new(c, K_A1, ashift(half, 7, 3)); S(c, x, t0); W(c, x, t1); SM(c, x, c->bnb);
        x = c->a0[l][2] = set_new(c, K_A2, ashift(quarter, 7, 3)); S(c, x, t1); W(c, x, a0img[l][0]); W(c, x, a0img[l][1]); SM(c, x, c->bnb);
        int T = c->a1n[l] = l == 0 ? 3 : 2;
        for (int t = 0; t < T; t++) {
            c->a1img[l][t][0] = img(c, quarter, RGBA8); c->a1img[l][t][1] = img(c, quarter, RGBA8);
            x = c->a1[l][t] = set_new(c, K_A3, ashift(quarter, 7, 3));
            S(c, x, a0img[l][0]); S(c, x, a0img[l][1]); W(c, x, c->a1img[l][t][0]); W(c, x, c->a1img[l][t][1]); SM(c, x, c->bnb);
        }
    }
    // Beta0 (over alpha1 of level 0, 3 temporal slots) and Beta1
    ext2 be = { (uint32_t)c->a1img[0][0][0].width, (uint32_t)c->a1img[0][0][0].height };
    id<MTLTexture> b0img[2] = { img(c, be, RGBA8), img(c, be, RGBA8) };
    for (int i = 0; i < 3; i++) {
        int x = c->b0[i] = set_new(c, K_B0, ashift(be, 7, 3));
        int o[3] = { (i + 1) % 3, (i + 2) % 3, i % 3 };
        for (int q = 0; q < 3; q++) { S(c, x, c->a1img[0][o[q]][0]); S(c, x, c->a1img[0][o[q]][1]); }
        W(c, x, b0img[0]); W(c, x, b0img[1]); SM(c, x, c->bnw);
    }
    id<MTLTexture> bt0[2] = { img(c, be, RGBA8), img(c, be, RGBA8) }, bt1[2] = { img(c, be, RGBA8), img(c, be, RGBA8) }, b1img[6];
    for (int i = 0; i < 6; i++) b1img[i] = img(c, shift(be, i), R8);
    {
        int x;
        x = c->b1[0] = set_new(c, K_B1, ashift(be, 7, 3)); S(c, x, b0img[0]); S(c, x, b0img[1]); W(c, x, bt0[0]); W(c, x, bt0[1]); SM(c, x, c->bnb);
        x = c->b1[1] = set_new(c, K_B2, ashift(be, 7, 3)); S(c, x, bt0[0]); S(c, x, bt0[1]); W(c, x, bt1[0]); W(c, x, bt1[1]); SM(c, x, c->bnb);
        x = c->b1[2] = set_new(c, K_B3, ashift(be, 7, 3)); S(c, x, bt1[0]); S(c, x, bt1[1]); W(c, x, bt0[0]); W(c, x, bt0[1]); SM(c, x, c->bnb);
        x = c->b1[3] = set_new(c, K_B4, ashift(be, 31, 5)); S(c, x, bt0[0]); S(c, x, bt0[1]); for (int i = 0; i < 6; i++) W(c, x, b1img[i]); SM(c, x, c->bnb); B(c, x, c->cb0);
    }
    // The generation pass (count = 1): gamma over all 7 levels (coarse to fine), delta over the 3 finest
    id<MTLTexture> g1img[7], d1img0[3], d1img1[3];
    for (int j = 0; j < 7; j++) {
        int l = 6 - j, T = c->a1n[l];
        ext2 e = { (uint32_t)c->a1img[l][0][0].width, (uint32_t)c->a1img[l][0][0].height };
        id<MTLTexture> g0img[3] = { img(c, e, RGBA8), img(c, e, RGBA8), img(c, e, RGBA8) };
        id<MTLTexture> gprev = j == 0 ? c->black : g1img[j - 1];
        for (int i = 0; i < T; i++) {   // Gamma0
            int x = c->g0[j][i] = set_new(c, K_G0, ashift(e, 7, 3)), p = (i + T - 1) % T, q = i % T;
            S(c, x, c->a1img[l][p][0]); S(c, x, c->a1img[l][p][1]); S(c, x, c->a1img[l][q][0]); S(c, x, c->a1img[l][q][1]); S(c, x, gprev);
            for (int k = 0; k < 3; k++) W(c, x, g0img[k]);
            SM(c, x, c->bnw); SM(c, x, c->eab); B(c, x, c->cb1);
        }
        {   // Gamma1
            id<MTLTexture> t0[2] = { img(c, e, RGBA8), img(c, e, RGBA8) }, t1[2] = { img(c, e, RGBA8), img(c, e, RGBA8) };
            g1img[j] = img(c, e, RGBA16F);
            id<MTLTexture> bet = b1img[j == 0 ? 5 : 6 - j];
            int x;
            x = c->g1[j][0] = set_new(c, K_G1, ashift(e, 7, 3)); for (int k = 0; k < 3; k++) S(c, x, g0img[k]); W(c, x, t0[0]); W(c, x, t0[1]); SM(c, x, c->bnb);
            x = c->g1[j][1] = set_new(c, K_G2, ashift(e, 7, 3)); S(c, x, t0[0]); S(c, x, t0[1]); W(c, x, t1[0]); W(c, x, t1[1]); SM(c, x, c->bnb);
            x = c->g1[j][2] = set_new(c, K_G3, ashift(e, 7, 3)); S(c, x, t1[0]); S(c, x, t1[1]); W(c, x, t0[0]); W(c, x, t0[1]); SM(c, x, c->bnb);
            x = c->g1[j][3] = set_new(c, K_G4, ashift(e, 7, 3)); S(c, x, t0[0]); S(c, x, t0[1]); S(c, x, gprev); S(c, x, bet); W(c, x, g1img[j]);
            SM(c, x, c->bnb); SM(c, x, c->eab); B(c, x, c->cb1);
        }
        if (j < 4) continue;
        int dj = j - 4;
        id<MTLTexture> dadd0 = dj == 0 ? c->black : d1img0[dj - 1], dadd1 = g1img[j - 1];
        id<MTLTexture> d0i0[3] = { img(c, e, RGBA8), img(c, e, RGBA8), img(c, e, RGBA8) }, d0i1 = img(c, e, RGBA8);
        for (int i = 0; i < T; i++) {   // Delta0
            int p = (i + T - 1) % T, q = i % T, x;
            x = c->d0s0[dj][i] = set_new(c, K_D0, ashift(e, 7, 3));
            S(c, x, c->a1img[l][p][0]); S(c, x, c->a1img[l][p][1]); S(c, x, c->a1img[l][q][0]); S(c, x, c->a1img[l][q][1]); S(c, x, dadd0);
            for (int k = 0; k < 3; k++) W(c, x, d0i0[k]);
            SM(c, x, c->bnw); SM(c, x, c->eab); B(c, x, c->cb1);
            x = c->d0s1[dj][i] = set_new(c, K_D5, ashift(e, 7, 3));
            S(c, x, c->a1img[l][p][0]); S(c, x, c->a1img[l][p][1]); S(c, x, c->a1img[l][q][0]); S(c, x, c->a1img[l][q][1]); S(c, x, dadd1); S(c, x, dadd0);
            W(c, x, d0i1); SM(c, x, c->bnw); SM(c, x, c->eab); B(c, x, c->cb1);
        }
        {   // Delta1
            id<MTLTexture> a0 = dj == 0 ? c->black : d1img0[dj - 1], a1 = b1img[6 - j], a2 = dj == 0 ? c->black : d1img1[dj - 1];
            id<MTLTexture> t0[2] = { img(c, e, RGBA8), img(c, e, RGBA8) }, t1[2] = { img(c, e, RGBA8), img(c, e, RGBA8) };
            d1img0[dj] = img(c, e, RGBA16F); d1img1[dj] = img(c, e, RGBA16F);
            int x, *d = c->d1[dj];
            x = d[0] = set_new(c, K_D1, ashift(e, 7, 3)); for (int k = 0; k < 3; k++) S(c, x, d0i0[k]); W(c, x, t0[0]); W(c, x, t0[1]); SM(c, x, c->bnb);
            x = d[1] = set_new(c, K_D2, ashift(e, 7, 3)); S(c, x, t0[0]); S(c, x, t0[1]); W(c, x, t1[0]); W(c, x, t1[1]); SM(c, x, c->bnb);
            x = d[2] = set_new(c, K_D3, ashift(e, 7, 3)); S(c, x, t1[0]); S(c, x, t1[1]); W(c, x, t0[0]); W(c, x, t0[1]); SM(c, x, c->bnb);
            x = d[3] = set_new(c, K_D4, ashift(e, 7, 3)); S(c, x, t0[0]); S(c, x, t0[1]); S(c, x, a0); S(c, x, a1); W(c, x, d1img0[dj]);
            SM(c, x, c->bnb); SM(c, x, c->eab); B(c, x, c->cb1);
            x = d[4] = set_new(c, K_D6, ashift(e, 7, 3)); S(c, x, d0i1); W(c, x, t0[0]); SM(c, x, c->bnb);
            x = d[5] = set_new(c, K_D7, ashift(e, 7, 3)); S(c, x, t0[0]); W(c, x, t1[0]); SM(c, x, c->bnb);
            x = d[6] = set_new(c, K_D8, ashift(e, 7, 3)); S(c, x, t1[0]); W(c, x, t0[0]); SM(c, x, c->bnb);
            x = d[7] = set_new(c, K_D9, ashift(e, 7, 3)); S(c, x, t0[0]); S(c, x, a2); W(c, x, d1img1[dj]);
            SM(c, x, c->bnb); SM(c, x, c->eab); B(c, x, c->cb1);
        }
    }
    for (int s = 0; s < 2; s++) {   // Generate: set 0 reads (second, first), set 1 (first, second)
        int x = c->gen[s] = set_new(c, K_GEN, ashift(c->sext, 15, 4));
        S(c, x, c->src[s ? 0 : 1]); S(c, x, c->src[s ? 1 : 0]); S(c, x, g1img[6]); S(c, x, d1img0[2]); S(c, x, d1img1[2]);
        W(c, x, c->out); SM(c, x, c->bnb); SM(c, x, c->eab); B(c, x, c->cb1);
    }
    return c;
}

id<MTLTexture> fg_source(fg_ctx *c) { return c->src[c->fidx % 2]; }
id<MTLTexture> fg_output(fg_ctx *c) { return c->out; }

static void run(id<MTLComputeCommandEncoder> e, const fg_set *s) {
    [e setComputePipelineState:s->k->pso];
    for (int i = 0; i < s->ns; i++) [e setTexture:s->s[i] atIndex:i];
    for (int i = 0; i < s->nw; i++) [e setTexture:s->w[i] atIndex:16 + i];
    for (int i = 0; i < s->nsm; i++) [e setSamplerState:s->sm[i] atIndex:i];
    if (s->b) [e setBuffer:s->b offset:0 atIndex:0];
    [e dispatchThreadgroups:s->groups threadsPerThreadgroup:s->k->tg];
}

void fg_encode(fg_ctx *c, id<MTLCommandBuffer> cb) {
    uint64_t f = c->fidx;
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    e.label = @"fg";
    run(e, &c->sets[c->mip[f % 2]]);
    for (int i = 0; i < 7; i++) {
        int l = 6 - i;
        for (int k = 0; k < 3; k++) run(e, &c->sets[c->a0[l][k]]);
        run(e, &c->sets[c->a1[l][f % c->a1n[l]]]);
    }
    run(e, &c->sets[c->b0[f % 3]]);
    for (int k = 0; k < 4; k++) run(e, &c->sets[c->b1[k]]);
    for (int j = 0; j < 7; j++) {
        int T = c->a1n[6 - j];
        run(e, &c->sets[c->g0[j][f % T]]);
        for (int k = 0; k < 4; k++) run(e, &c->sets[c->g1[j][k]]);
        if (j < 4) continue;
        run(e, &c->sets[c->d0s0[j - 4][f % T]]);
        run(e, &c->sets[c->d0s1[j - 4][f % T]]);
        for (int k = 0; k < 8; k++) run(e, &c->sets[c->d1[j - 4][k]]);
    }
    run(e, &c->sets[c->gen[f % 2]]);
    [e endEncoding];
    c->fidx++;
}

void fg_destroy(fg_ctx *c) {
    if (!c) return;
    for (int i = 0; i < K_N; i++) c->k[i].pso = nil;
    c->keep = nil; c->dev = nil; c->bnb = c->bnw = c->eab = nil; c->cb0 = c->cb1 = nil;
    c->src[0] = c->src[1] = c->out = c->black = nil;
    free(c->sets); free(c);
}
