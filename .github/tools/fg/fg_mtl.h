// SPDX-License-Identifier: GPL-3.0-or-later
// Frame generation on Metal: a port of lsfg-vk's chain, performance or quality mode (lsfg-vk-backend, GPL-3.0-or-later),
// running the compute shaders prep.py converts from the user's own DLL. One generated frame between each pair.
//
//   fg_ctx *c = fg_create(device, dir, w, h, 1.0f / flow_scale, &err);   // dir holds <role>.metallib + manifest.json
//   each real frame: blit it into fg_source(c), then fg_encode(c, cmdbuf): the frame between the previous and this one
//   is in fg_output(c) once cmdbuf completes. Show fg_output, then the real frame.
#pragma once
#import <Metal/Metal.h>

typedef struct fg_ctx fg_ctx;

fg_ctx *fg_create(id<MTLDevice> dev, NSString *shaderDir /* nil: the shaders built in (FG_EMBEDDED) */, uint32_t width, uint32_t height, float flow,
                  int quality /* 0: performance chain, 1: quality chain */, NSString **err);
id<MTLTexture> fg_source(fg_ctx *c);    // RGBA8Unorm, width x height: write the newest real frame here before fg_encode
id<MTLTexture> fg_output(fg_ctx *c);    // RGBA8Unorm: the generated frame after fg_encode's work completes
void fg_encode(fg_ctx *c, id<MTLCommandBuffer> cb);
// Two pictures between the pair (a third and two thirds of the way), for a game frame shown for three refreshes
void fg_encode_n(fg_ctx *c, id<MTLCommandBuffer> cb, int n /* 1 or 2 */);
id<MTLTexture> fg_output_n(fg_ctx *c, int i);
void fg_destroy(fg_ctx *c);
