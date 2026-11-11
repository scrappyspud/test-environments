// Test for fg_mtl on a Mac: synthetic frames (a textured background, a static HUD bar, a red disc moving 40 px a frame),
// fed one by one; after each frame the generated frame should show the disc halfway between the last two positions.
// Prints the disc's centroid in each generated frame and the GPU time per frame; writes PNGs of the last pair and output.
// fgtest <shader dir> <out dir> [width height flow_scale [mode: 1 quality, 2 two pictures per pair, 3 both]]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include <math.h>
#include "fg_mtl.h"

static uint32_t W = 1280, H = 896;
static void draw(uint8_t *p, double cx) {
    for (uint32_t y = 0; y < H; y++) for (uint32_t x = 0; x < W; x++) {
        uint8_t *q = p + 4 * (y * W + x);
        double n = 0.5 + 0.25 * sin(x * 0.07) * cos(y * 0.05) + 0.25 * sin((x + 2 * y) * 0.013);   // static texture
        q[0] = (uint8_t)(40 + 90 * n); q[1] = (uint8_t)(70 + 120 * n); q[2] = (uint8_t)(60 + 60 * (1 - n)); q[3] = 255;
        if (y < 48) { q[0] = q[1] = q[2] = 230; }                                                   // HUD bar
        double dx = x - cx, dy = y - H * 0.55;
        if (dx * dx + dy * dy < 50.0 * 50.0) { q[0] = 230; q[1] = 30; q[2] = 30; }                  // the disc
    }
}
static double centroid(const uint8_t *p, int *count) {
    double sx = 0; int n = 0;
    for (uint32_t y = 48; y < H; y++) for (uint32_t x = 0; x < W; x++) {
        const uint8_t *q = p + 4 * (y * W + x);
        if (q[0] > 170 && q[1] < 90 && q[2] < 90) { sx += x; n++; }
    }
    *count = n; return n ? sx / n : -1;
}
static void png(NSString *path, const uint8_t *p) {
    CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
    CGContextRef ctx = CGBitmapContextCreate((void *)p, W, H, 8, 4 * W, cs, kCGImageAlphaNoneSkipLast);
    CGImageRef im = CGBitmapContextCreateImage(ctx);
    CGImageDestinationRef d = CGImageDestinationCreateWithURL((__bridge CFURLRef)[NSURL fileURLWithPath:path], CFSTR("public.png"), 1, NULL);
    CGImageDestinationAddImage(d, im, NULL); CGImageDestinationFinalize(d);
    CFRelease(d); CGImageRelease(im); CGContextRelease(ctx); CGColorSpaceRelease(cs);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 3) { fprintf(stderr, "usage: fgtest <shader dir> <out dir> [w h flow_scale]\n"); return 2; }
        float flowScale = 1.0f;
        if (argc >= 6) { W = atoi(argv[3]); H = atoi(argv[4]); flowScale = atof(argv[5]); }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", dev.name.UTF8String);
        NSString *err = nil;
        int quality = argc >= 7 && atoi(argv[6]) & 1, two = argc >= 7 && atoi(argv[6]) & 2;   // 7th argument: 1 quality chain, 2 two pictures per pair, 3 both
        printf("%s chain, %s\n", quality ? "quality" : "performance", two ? "two pictures per pair" : "one picture per pair");
        fg_ctx *c = fg_create(dev, strcmp(argv[1], "-") ? @(argv[1]) : nil, W, H, 1.0f / flowScale, quality, &err);
        id<MTLBuffer> rb2 = [dev newBufferWithLength:4 * W * H options:MTLResourceStorageModeShared];
        if (!c) { printf("FAIL create: %s\n", err.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        size_t sz = 4 * W * H; uint8_t *frame = malloc(sz), *prev = malloc(sz), *out = malloc(sz);
        MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm width:W height:H mipmapped:NO];
        d.storageMode = MTLStorageModeShared; d.usage = MTLTextureUsageShaderRead;
        id<MTLTexture> up = [dev newTextureWithDescriptor:d];
        id<MTLBuffer> rb = [dev newBufferWithLength:sz options:MTLResourceStorageModeShared];
        int ok = 1; double gpuSum = 0; int gpuN = 0;
        for (int t = 0; t < 10; t++) {
            double cx = 200 + 40.0 * t;
            memcpy(prev, frame, sz); draw(frame, cx);
            [up replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:frame bytesPerRow:4 * W];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
            [b copyFromTexture:up toTexture:fg_source(c)]; [b endEncoding];
            if (two) fg_encode_n(c, cb, 2); else fg_encode(c, cb);
            b = [cb blitCommandEncoder];
            if (two) [b copyFromTexture:fg_output_n(c, 1) sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1)
                      toBuffer:rb2 destinationOffset:0 destinationBytesPerRow:4 * W destinationBytesPerImage:sz];
            [b copyFromTexture:fg_output(c) sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, H, 1)
                      toBuffer:rb destinationOffset:0 destinationBytesPerRow:4 * W destinationBytesPerImage:sz];
            [b endEncoding];
            [cb commit]; [cb waitUntilCompleted];
            if (cb.error) { printf("FAIL frame %d: %s\n", t, cb.error.localizedDescription.UTF8String); return 1; }
            double ms = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
            memcpy(out, rb.contents, sz);
            int n; double gx = centroid(out, &n);
            uint64_t sum = 0; for (size_t i = 0; i < sz; i += 4) sum += out[i] + out[i + 1] + out[i + 2];
            double want = two ? cx - 40 + 40.0 / 3 : cx - 20;
            printf("frame %d: real disc at %.0f, generated disc at %.1f (%d px; expected %.0f), mean %.1f, GPU %.2f ms\n",
                   t, cx, gx, n, want, sum / (3.0 * W * H), ms);
            if (t >= 3) { gpuSum += ms; gpuN++; if (n < 2000 || fabs(gx - want) > 12) ok = 0; }
            if (two) { int n2; double g2 = centroid(rb2.contents, &n2), want2 = cx - 40 + 80.0 / 3;
                printf("         second picture: disc at %.1f (%d px; expected %.0f)\n", g2, n2, want2);
                if (t >= 3 && (n2 < 2000 || fabs(g2 - want2) > 12)) ok = 0; }
            if (t == 6) { NSString *o = @(argv[2]);
                png([o stringByAppendingPathComponent:@"a_prev.png"], prev); png([o stringByAppendingPathComponent:@"c_next.png"], frame);
                png([o stringByAppendingPathComponent:@"b_generated.png"], out); }
        }
        printf("GPU per generated frame (incl. two blits): %.2f ms average over the last %d\n", gpuSum / gpuN, gpuN);
        if (!ok) { printf("CHECK: the generated disc is not where expected (see PNGs)\n"); return 3; }
        printf(ok ? "PASS: the generated disc sits between the real ones\n" : "CHECK: the generated disc is not where expected (see PNGs)\n");
        fg_destroy(c);
        return 0;
    }
}
