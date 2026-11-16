#!/usr/bin/env python3
# shape.py <nxnative checkout>: a 4:3 picture for one game build (main b424be15...), so a 4:3-ish screen has (almost) no black bars. Three parts, all for that build only:
#   camera: one registered cut (the game's own aspect re-set before each projection update is skipped, so its constructed 4:3 projection stays), original bytes asserted
#   present: the 16:9 raster is shown in a 4:3 box (the game draws its 4:3 view into the 16:9 buffer; the screen stretches it back); DM_PRESENT_SHAPE=16:9 in env.txt turns it off
#   HUD: shape_img.py corrects the layout archives inside the game file image (RomFS.img) right after extraction, so the image, its identity and Game Drive all stay as usual
# Any other game: the cut's build ID never matches, the present check names one profile, and shape_img.py leaves other builds' images alone.
import sys, os, shutil
root = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
TITLE = 'PROFILE'

# camera cut (the game's profile has no patches.py upstream; stop if one appears, it would need merging)
p = os.path.join(root, 'dm/games', TITLE, 'patches.py')
assert os.path.isdir(os.path.dirname(p)), 'game profile missing upstream'
assert not os.path.exists(p), 'upstream now has a patches.py for this game: merge the cut into it'
open(p, 'w').write('''# shape.py: 4:3 picture. 1.4.1 main: the projection's aspect re-set (bl at main+0x73de88) is skipped, keeping the constructed 4:3 aspect (as the community narrow-aspect mods do)
CUTS = {
  'b424be150a8e7d78701cbe7a439d9ebf': [
    (0x73de88, '1aa90894', '1f2003d5', 'projection aspect re-set: bl -> nop (4:3 view, shape.py)'),
  ],
}
''')

# present: 4:3 box for this profile
p = os.path.join(root, 'dm/gfx/nvn/nvn_mtl.m'); s = open(p).read()
a = '''static void present_shape(id<MTLTexture> src, double *w, double *h) {
'''
assert s.count(a) == 1, 'present_shape moved upstream'
s = s.replace(a, a + '''    {   // shape.py: this game's 4:3 view, drawn into its 16:9 buffer, shown in a 4:3 box; DM_PRESENT_SHAPE=W:H overrides (16:9 = the game's own)
        static double fw = -1, fh;
        if (fw < 0) {
            fw = 0; unsigned a = 0, b = 0; const char *e = getenv("DM_PRESENT_SHAPE");
            if (e && sscanf(e, "%u:%u", &a, &b) == 2 && a && b) { fw = a; fh = b; }
            else if (!strcmp(DM_GAME_ID, "''' + TITLE + '''")) { fw = 4; fh = 3; }
            if (fw > 0) MLOG("present: picture shape %.0f:%.0f (shape.py)", fw, fh);
        }
        if (fw > 0) { *w = fw; *h = fh; return; }
    }
''', 1)
open(p, 'w').write(s)

# HUD: correct the layouts inside each extracted image, before its format check (whose stamp keeps the image's size and time)
shutil.copy(os.path.join(here, 'shape_img.py'), os.path.join(root, 'scripts/shape_img.py'))
p = os.path.join(root, 'scripts/build_ipa_from_nsp.py'); s = open(p).read()
a = '    check_formats(dirs)\n'
assert s.count(a) == 1, 'check_formats call moved upstream'
s = s.replace(a, '''    for _d in dirs.values():   # shape.py: 4:3 layouts for its one build (other builds untouched)
        if (Path(_d) / 'RomFS.img').is_file() and subprocess.run([sys.executable, '-I', str(Path(__file__).resolve().parent / 'shape_img.py'), str(_d)]).returncode: raise BuildError('shape_img.py failed')
''' + a, 1)
open(p, 'w').write(s)
print('4:3 shape added (camera cut, present box, HUD layouts)')
