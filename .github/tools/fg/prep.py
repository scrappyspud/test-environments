#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Shader prep for the frame generation module: reads the compute shaders (SPIR-V) out of the user's own DLL,
# converts the performance-mode chain to MSL with SPIRV-Cross, remaps the Vulkan bindings to Metal slots and
# writes one .metal per role plus a manifest. Usage: prep.py <dll> <outdir> [--fp32]
# Bindings (lsfg-vk layout): uniform buffers 0.., samplers 16.., sampled images 32.., storage images 48..
#   -> Metal: buffer(i), sampler(i), texture(i) for sampled, texture(16+i) for storage.
import sys, os, re, json, struct, subprocess, pefile

def resources(dll):
    pe = pefile.PE(dll, fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']])
    out = {}
    for t in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        if t.id != 10: continue
        for e in t.directory.entries:
            for l in e.directory.entries:
                out[e.id] = pe.get_data(l.data.struct.OffsetToData, l.data.struct.Size)
    return out

# role: (shader id, performance variant, sampled, storage, buffers, samplers)  (counts as in lsfg-vk's registry, performance mode)
ROLES = {
    'mipmaps': (255, False, 1, 7, 1, 1), 'generate': (256, False, 5, 1, 1, 2),
    'alpha0': (267, True, 1, 1, 0, 1), 'alpha1': (268, True, 1, 1, 0, 1), 'alpha2': (269, True, 1, 2, 0, 1), 'alpha3': (270, True, 2, 2, 0, 1),
    'beta0': (275, True, 6, 2, 0, 1), 'beta1': (276, True, 2, 2, 0, 1), 'beta2': (277, True, 2, 2, 0, 1), 'beta3': (278, True, 2, 2, 0, 1), 'beta4': (279, True, 2, 6, 1, 1),
    'gamma0': (257, True, 5, 3, 1, 2), 'gamma1': (259, True, 3, 2, 0, 1), 'gamma2': (260, True, 2, 2, 0, 1), 'gamma3': (261, True, 2, 2, 0, 1), 'gamma4': (262, True, 4, 1, 1, 2),
    'delta0': (257, True, 5, 3, 1, 2), 'delta1': (263, True, 3, 2, 0, 1), 'delta2': (264, True, 2, 2, 0, 1), 'delta3': (265, True, 2, 2, 0, 1), 'delta4': (266, True, 4, 1, 1, 2),
    'delta5': (258, True, 6, 1, 1, 2), 'delta6': (271, True, 1, 1, 0, 1), 'delta7': (272, True, 1, 1, 0, 1), 'delta8': (273, True, 1, 1, 0, 1), 'delta9': (274, True, 2, 1, 1, 2),
}

def patch_generate(code, hdr=False):   # as lsfg-vk: drop StorageImageWriteWithoutFormat, give the output image a format
    w = list(struct.unpack('<%dI' % (len(code) // 4), code)); i = 5
    while i < len(w):
        wc, op = w[i] >> 16, w[i] & 0xffff
        if op == 17 and wc >= 2 and w[i + 1] == 56: w[i + 1] = 1
        if op == 25 and wc >= 9 and w[i + 7] == 2: w[i + 8] = 2 if hdr else 4
        i += wc or 1
    return struct.pack('<%dI' % len(w), *w)

def local_size(code):
    w = struct.unpack('<%dI' % (len(code) // 4), code); i = 5
    while i < len(w):
        wc, op = w[i] >> 16, w[i] & 0xffff
        if op == 16 and w[i + 2] == 17: return list(w[i + 3:i + 6])   # OpExecutionMode LocalSize
        i += wc or 1
    raise SystemExit('no LocalSize')

def entry_name(code):
    w = struct.unpack('<%dI' % (len(code) // 4), code); i = 5
    while i < len(w):
        wc, op = w[i] >> 16, w[i] & 0xffff
        if op == 15: return struct.pack('<%dI' % (wc - 3), *w[i + 3:i + wc]).split(b'\0')[0].decode()   # OpEntryPoint
        i += wc or 1
    raise SystemExit('no entry point')

def remap(src):
    src = re.sub(r'\[\[texture\((\d+)\)\]\]', lambda m: '[[texture(%d)]]' % (int(m[1]) - 32 if int(m[1]) < 48 else int(m[1]) - 48 + 16), src)
    return re.sub(r'\[\[sampler\((\d+)\)\]\]', lambda m: '[[sampler(%d)]]' % (int(m[1]) - 16), src)

def main():
    dll, out = sys.argv[1], sys.argv[2]; fp16 = '--fp32' not in sys.argv
    os.makedirs(out, exist_ok=True); res = resources(dll); man = {}
    xc = os.environ.get('SPIRV_CROSS', 'spirv-cross')
    for role, (sid, perf, ns, nst, nb, nsm) in ROLES.items():
        rid = 49 + sid + (23 if perf else 0) + (0 if fp16 else 49)
        code = res.get(rid)
        if code is None or code[:4] != b'\x03\x02\x23\x07': raise SystemExit(f'{role}: resource {rid} missing or not SPIR-V')
        if role == 'generate': code = patch_generate(code)
        spv = os.path.join(out, role + '.spv'); open(spv, 'wb').write(code)
        r = subprocess.run([xc, '--msl', '--msl-version', '20300', '--msl-decoration-binding', '--rename-entry-point', entry_name(code), 'k_' + role, 'comp', spv],
                           capture_output=True, text=True)
        if r.returncode: raise SystemExit(f'{role}: spirv-cross failed: {r.stderr[:300]}')
        msl = remap(r.stdout); os.remove(spv)
        tex = [int(t) for t in re.findall(r'\[\[texture\((\d+)\)\]\]', msl)]
        got = (sum(t < 16 for t in tex), sum(t >= 16 for t in tex), len(re.findall(r'\[\[buffer\(', msl)), len(re.findall(r'\[\[sampler\(', msl)))
        if got != (ns, nst, nb, nsm): print(f'note {role}: bindings {got} vs registry {(ns, nst, nb, nsm)} (unused ones are dropped by SPIRV-Cross)')
        open(os.path.join(out, role + '.metal'), 'w').write(msl)
        man[role] = {'tg': local_size(code), 'fp16': fp16}
    json.dump(man, open(os.path.join(out, 'manifest.json'), 'w'), indent=1)
    print(f'{len(man)} shaders converted ({"fp16" if fp16 else "fp32"})')

main()
