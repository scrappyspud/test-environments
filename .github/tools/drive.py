#!/usr/bin/env python3
# drive.py <nxnative checkout>: "game drive" for the iOS app. The game files (RomFS.img) can stay on a USB drive, like a cartridge:
# with no usable game files in the app, it offers "Choose Game Drive Folder…" (the system's folder picker); the folder is kept as a
# security-scoped bookmark (the same way CloudSaves.m keeps the cloud-save folder) and read from there at every launch while the drive
# is plugged in. Game files in Documents still come first when a drive folder is set but holds no usable image (drive unplugged).
# Saves, the device's pipeline archive and env.txt stay in Documents.
import sys, os
root = sys.argv[1]
S = os.path.join(root, 'dm/ios/Sources')

open(os.path.join(S, 'GameDrive.h'), 'w').write(r'''// Game drive: the folder on a USB drive (or anywhere the Files app reaches) that holds this game's RomFS.img, picked once and
// kept as a security-scoped bookmark (bound to the app's signature: a re-signed build asks for the folder again).
#import <Foundation/Foundation.h>
NSString *dm_game_drive_path(BOOL *stale);   // the folder, its scope held for the session; nil when none is set or it does not resolve
BOOL dm_store_game_drive(NSURL *url);        // keep the folder the player just picked
void dm_clear_game_drive(void);
int dm_game_drive_set(void);                 // a folder was picked (resolving or not)
''')

open(os.path.join(S, 'GameDrive.m'), 'w').write(r'''// See GameDrive.h. Modelled on CloudSaves.m.
#import "GameDrive.h"
static NSString *const kGameDrive = @"gameDriveBookmark";
static NSURL *opened = nil;   // held for the session: the host reads the image for as long as the game runs
static NSData *fresh_bookmark(NSURL *url) {
    BOOL scoped = [url startAccessingSecurityScopedResource];
    NSData *b = [url bookmarkDataWithOptions:0 includingResourceValuesForKeys:nil relativeToURL:nil error:nil];
    if (scoped) [url stopAccessingSecurityScopedResource];
    return b;
}
int dm_game_drive_set(void) { return [NSUserDefaults.standardUserDefaults dataForKey:kGameDrive].length > 0; }
NSString *dm_game_drive_path(BOOL *stale) {
    if (stale) *stale = NO;
    NSData *b = [NSUserDefaults.standardUserDefaults dataForKey:kGameDrive];
    if (b.length == 0) return nil;
    BOOL isStale = NO; NSError *e = nil;
    NSURL *url = [NSURL URLByResolvingBookmarkData:b options:0 relativeToURL:nil bookmarkDataIsStale:&isStale error:&e];
    if (!url) {
        fprintf(stderr, "boot: the game drive folder does not resolve (%s): drive unplugged, or the app is signed differently now\n", e.localizedDescription.UTF8String ?: "no reason given");
        if (stale) *stale = YES;
        return nil;
    }
    if (opened && ![opened.path isEqualToString:url.path]) opened = nil;
    if (!opened) [url startAccessingSecurityScopedResource];
    opened = url;
    if (isStale) { NSData *again = fresh_bookmark(url); if (again) [NSUserDefaults.standardUserDefaults setObject:again forKey:kGameDrive]; }
    return url.path;
}
BOOL dm_store_game_drive(NSURL *url) {
    NSData *b = fresh_bookmark(url); NSError *e = nil; NSURL *back = nil;
    if (b) back = [NSURL URLByResolvingBookmarkData:b options:0 relativeToURL:nil bookmarkDataIsStale:nil error:&e];
    if (!b || !back) { fprintf(stderr, "[RED] could not bookmark the game drive folder %s: %s\n", url.path.UTF8String ?: "?", e.localizedDescription.UTF8String ?: "no bookmark"); return NO; }
    [NSUserDefaults.standardUserDefaults setObject:b forKey:kGameDrive];
    [NSUserDefaults.standardUserDefaults synchronize];
    fprintf(stderr, "launcher: game drive folder set to %s\n", url.path.UTF8String ?: "?");
    return YES;
}
void dm_clear_game_drive(void) { [NSUserDefaults.standardUserDefaults removeObjectForKey:kGameDrive]; [NSUserDefaults.standardUserDefaults synchronize]; }
''')

# main.m: look on the game drive first
p = os.path.join(S, 'main.m'); s = open(p).read()
a = '#include "front.h"\n'
assert s.count(a) == 1, 'main.m includes moved'
s = s.replace(a, a + '#import "GameDrive.h"\n', 1)
a = '''    for (NSString *img in romfs_document_images(doc)) {
        if (!romfs_image_complete(img) || !romfs_identity_accepts(img)) continue;'''
assert s.count(a) == 1, 'romfs_locate moved upstream'
s = s.replace(a, '''    {   // drive.py: the game drive (GameDrive.m) first, when a folder was picked and the drive is plugged in
        BOOL stale = NO; NSString *gd = dm_game_drive_path(&stale);
        if (gd) {
            for (NSString *img in romfs_document_images(gd)) if (romfs_image_complete(img) && romfs_identity_accepts(img)) { fprintf(stderr, "boot: game files from the game drive\\n"); return img; }
            fprintf(stderr, "boot: the game drive folder has no usable RomFS.img (another game's, or still copying); looking in Documents\\n");
        } else if (dm_game_drive_set()) fprintf(stderr, "boot: game drive not reachable (unplugged?); looking in Documents\\n");
    }
''' + a, 1)
open(p, 'w').write(s)

# front_ios.m: offer the folder picker when the game files are missing
p = os.path.join(S, 'front_ios.m'); s = open(p).read()
a = '#import "CloudSaves.h"\n'
assert s.count(a) == 1
s = s.replace(a, a + '#import "GameDrive.h"\n#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>\n', 1)
a = '@interface DmRootVC : UIViewController'
assert s.count(a) == 1
s = s.replace(a, r'''// drive.py: the folder picker for the game drive, and what to say after it
@interface DmDrivePick : NSObject <UIDocumentPickerDelegate>
@property (weak) UIViewController *vc;
@end
@implementation DmDrivePick
- (void)documentPicker:(UIDocumentPickerViewController *)c didPickDocumentsAtURLs:(NSArray<NSURL *> *)urls {
    NSURL *u = urls.firstObject; BOOL ok = u && dm_store_game_drive(u);
    UIAlertController *a = [UIAlertController alertControllerWithTitle:ok ? @"Game drive set" : @"Could not use that folder"
        message:ok ? @"Close the app completely (swipe it away) and open it again. With the drive plugged in, the game reads its files from there." : @"Pick the folder that holds this game's RomFS.img."
        preferredStyle:UIAlertControllerStyleAlert];
    [a addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
    [self.vc presentViewController:a animated:YES completion:nil];
}
@end
static DmDrivePick *g_drive_pick;
static void dm_drive_alert(UIViewController *vc, const char *why) {
    NSString *m = [NSString stringWithFormat:@"%@\n\nKeep RomFS.img in this app's Documents, or on a USB drive: choose the drive folder that holds it.%@",
        why ? @(why) : @"Game files not found.", dm_game_drive_set() ? @"\n\nA game drive folder is set: is the drive plugged in?" : @""];
    UIAlertController *a = [UIAlertController alertControllerWithTitle:@"Game files" message:m preferredStyle:UIAlertControllerStyleAlert];
    [a addAction:[UIAlertAction actionWithTitle:@"Choose Game Drive Folder…" style:UIAlertActionStyleDefault handler:^(UIAlertAction *x) {
        UIDocumentPickerViewController *p = [[UIDocumentPickerViewController alloc] initForOpeningContentTypes:@[UTTypeFolder]];
        g_drive_pick = [DmDrivePick new]; g_drive_pick.vc = vc; p.delegate = g_drive_pick;
        [vc presentViewController:p animated:YES completion:nil];
    }]];
    if (dm_game_drive_set()) [a addAction:[UIAlertAction actionWithTitle:@"Forget the Game Drive" style:UIAlertActionStyleDestructive handler:^(UIAlertAction *x) { dm_clear_game_drive(); }]];
    [a addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleCancel handler:nil]];
    [vc presentViewController:a animated:YES completion:nil];
}
''' + a, 1)
a = '''    if (!front_ios_boot_ok() && front_ios_boot_message() && !self.presentedViewController) {   // a slim build whose game file image is missing or another version (main.m)
        UIAlertController *a = [UIAlertController alertControllerWithTitle:@"Game files" message:@(front_ios_boot_message()) preferredStyle:UIAlertControllerStyleAlert];
        [a addAction:[UIAlertAction actionWithTitle:@"OK" style:UIAlertActionStyleDefault handler:nil]];
        [self presentViewController:a animated:YES completion:nil];
    }'''
assert s.count(a) == 1, 'viewDidAppear alert moved upstream'
s = s.replace(a, '''    if (!front_ios_boot_ok() && !self.presentedViewController) dm_drive_alert(self, front_ios_boot_message());   // drive.py: no usable game files: say why, offer the game drive''', 1)
open(p, 'w').write(s)
print('game drive added')
