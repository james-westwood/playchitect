# GUI Design — Navigation Sidebar Layout

*Status: Planned (Milestone 7)*
*Last updated: 2026-02-23*

## Design decisions

### Navigation model

Three options were considered:

| Approach | Decision |
|---|---|
| Full menu bar | ✗ Dated for Adwaita; features don't map cleanly to menu hierarchy |
| Tabs | ~ Works for 3 views, but Set Builder and Playlists are too rich, and the sequential workflow doesn't suit peer tabs |
| **Navigation sidebar** | ✓ GNOME HIG compliant, scales to all planned features, collapses gracefully, separates navigation from in-view controls |

The sidebar follows the `Adw.OverlaySplitView` pattern used by GNOME Music, Nautilus, and Fractal. It collapses to a hamburger button below ~700px.

### Four views

The app is structured around four user jobs-to-be-done:

```
Library  →  Playlists  →  Set Builder  →  Export
 (what      (sort it)      (build a set     (get it
 have I                    interactively)    out)
  got?)
```

### App-level actions

Persistent settings and housekeeping live in the header's `Adw.MenuButton` (⋮):

- Open Music Folder…
- Preferences… (`Adw.PreferencesWindow`)
- Keyboard Shortcuts
- About Playchitect

Preferences holds stable settings: library paths, default export format, DJ software paths (Mixxx DB, Rekordbox XML), keybindings.

---

## Overall layout — #112

```
┌─────────────────────────────────────────────────────────────────┐
│  [≡]  Playchitect                        [view-specific btns] ⋮ │
├────────────┬────────────────────────────────────────────────────┤
│            │                                                     │
│  Library   │                                                     │
│            │         Active view content                        │
│  Playlists │                                                     │
│            │                                                     │
│  Set       │                                                     │
│  Builder   │                                                     │
│            │                                                     │
│  Export    │                                                     │
│            │                                                     │
└────────────┴────────────────────────────────────────────────────┘
```

**Implementation**: `Adw.OverlaySplitView`, sidebar ~180px wide, `navigation-sidebar` CSS class on the `Gtk.ListBox` rows.

---

## Library view — #113

The foundation. Browse the full track collection, scan new folders, preview individual tracks.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  [≡]  Library                  [📁 Open Folder]  [🔍]  [◨ Preview]  ⋮  │
├────────────────────────────────────────────────┬────────────────────────┤
│  🔍 Search tracks…              [Format ▾]     │                        │
│  ──────────────────────────────────────────    │  ┌──────────────────┐  │
│  Title              Artist     BPM  ████  Dur  │  │                  │  │
│  ──────────────────────────────────────────    │  │   Cover art      │  │
│  Strings of Life    D. May     128  ████  8:22 │  │   240 × 240      │  │
│  Can You Feel It    Fingers    126  ███   6:11 │  │                  │  │
│  Nude Photo      ●  Model 500  122  █████ 7:44 │  └──────────────────┘  │
│  The Bells          J. Saul    133  ██    5:02 │                        │
│  Jaguar             C. Craig   128  ████  9:15 │  Nude Photo            │
│  Nos Amis           LFO        136  █████ 6:30 │  Model 500             │
│  …                                             │  Nude Photo EP · 1987  │
│                                                │                        │
│  2,147 tracks                                  │  BPM   Key   Dur  Fmt  │
│                                                │  122   9A    7:44 FLAC │
│                                                │                        │
│                                                │  ◄   ▶/▐▐   ►          │
│                                                │  ──────────○─────────  │
│                                                │  2:11 / 7:44   🔊 ─○  │
└────────────────────────────────────────────────┴────────────────────────┘
```

### Track list (left pane)
- Columns: Title, Artist, BPM, Intensity bar, Duration — expandable later with Key, Mood, Tags
- Click column header to sort; search filters in real time
- Format filter chip: All / FLAC / MP3 / etc.
- Track count at bottom-left

### Preview panel (right pane) — #114
Collapsible via `[◨ Preview]` toggle. Auto-opens on first track selection, then respects user's last state.

- **Cover art** (240×240): extracted from embedded tags (mutagen), falling back to `cover.jpg`/`folder.jpg` in the same directory, then a placeholder SVG
- **Metadata**: title, artist, album, year, BPM, key, duration, format
- **Embedded audio player** using GStreamer (`gi.repository.Gst`, `playbin` element):
  - Play/pause toggle, skip ±15 s, seek bar, volume knob
  - Position updated via `GLib.timeout_add(200ms)`
  - Spacebar shortcut triggers play/pause
- Replaces the Sushi/xdg-open integration entirely — no external app dependency

---

## Playlists view — #115

Configure a clustering run, browse results, inspect each cluster's sequenced track list.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  [≡]  Playlists  [▶ Cluster]  Size:[20 ▾]  Arc:[Ramp ▾]  [⚙ Weights]  │
├──────────────────────┬──────────────────────────────────────────────────┤
│                      │  Deep Techno — 14 tracks — 112 min               │
│  ♦ Deep Techno       │  ─────────────────────────────────────────────── │
│    128–132 BPM       │  ▁▁▂▃▄▅▆▇▇█▇▆▅▄▃▂▁   ← energy arc              │
│    ████████  intens  │  ─────────────────────────────────────────────── │
│    14 tracks  112min │  Title              Artist     BPM  ████  Dur    │
│                      │  1  Strings of Life  D. May    128  ████  8:22   │
│  ♦ Ambient Techno    │  2  Can You Feel It  Fingers   126  ███   6:11   │
│    90–100 BPM        │  3  Nude Photo       Model 500 122  █████ 7:44   │
│    ███       intens  │  4  The Bells        J. Saul   133  ██    5:02   │
│    11 tracks   82min │  5  Jaguar           C. Craig  128  ████  9:15   │
│                      │  …                                               │
│  ♦ Hard Breaks       │                                                   │
│    140–148 BPM       │                                                   │
│    ██████    intens  │                                                   │
│    9 tracks    67min │                                                   │
│                      │                                                   │
│  ♦ Peak Time         │                                                   │
│    138–145 BPM       │                                                   │
│    █████████ intens  │                                                   │
└──────────────────────┴──────────────────────────────────────────────────┘
```

### Inline controls (header bar)
| Control | Behaviour |
|---|---|
| `[▶ Cluster]` | Runs intensity analysis + clustering in background |
| `Size: [20 ▾]` | Target tracks per playlist: 10 / 20 / 30 / 45 / 60 / Custom |
| `Arc: [Ramp ▾]` | Sequencing mode: Ramp / Peak / Valley / Wave / Flat |
| `[⚙ Weights]` | Popover with per-feature weight sliders |

### Cluster cards (left panel)
Name (editable, auto-named by #104), BPM range, intensity bar, track count, total duration.

### Track list (right panel)
- Energy arc sparkline across the top — hardness per track in sequence order
- Numbered rows, drag-to-reorder (prerequisite for Set Builder)
- Columns: #, Title, Artist, BPM, Intensity bar, Duration

---

## Set Builder view — #101

Interactive track-by-track set construction. Distinct from the batch clustering in Playlists — here the user is building a specific set with full creative control.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  [≡]  Set Builder   Mode:[5 Rhythms ▾]   Target:[90 min]   [🎲 Auto]   │
├─────────────────────────────────┬───────────────────────────────────────┤
│  Your set                 61min │  Suggestions                          │
│  ─────────────────────────────  │  ───────────────────────────────────  │
│  ▁▂▃▅▆▇█▇▆▅▄▃▂▁  energy arc   │  Compatible next tracks:              │
│  ─────────────────────────────  │                                       │
│  1  Strings of Life  128  11A ██│  ♦ Jaguar      128   8A  ████  9:15  │
│  2  Can You Feel It  126   6A █ │  ♦ Nude Photo  122   9A  █████ 7:44  │
│  3  Jaguar           128   8A ██│  ♦ The Bells   133  10A  ██    5:02  │
│  4  ▸ Drop track here…          │                                       │
│                                 │  ┌── Camelot wheel ────────────────┐  │
│                                 │  │         11A                     │  │
│                                 │  │       ╱     ╲                   │  │
│                                 │  │     8A   ─   6A                 │  │
│                                 │  │       ╲     ╱                   │  │
│                                 │  │         9A                      │  │
│                                 │  └─────────────────────────────────┘  │
└─────────────────────────────────┴───────────────────────────────────────┘
```

### Left pane — the set
- Ordered track list with live energy arc above
- Drag-to-reorder
- Duration counter
- `[🎲 Auto]` populates from best-matching library tracks for the chosen mode

### Right pane — suggestions
- Compatible next-track suggestions based on harmonic key, BPM, and energy (#100)
- Camelot wheel diagram highlighting current key and compatible neighbours (#36)
- 5 Rhythms mode constrains suggestions to the correct rhythmic phase (#51)

---

## Export view — #116

```
┌─────────────────────────────────────────────────────────────────────────┐
│  [≡]  Export                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Format                                                                  │
│  ──────────────────────────────────────────────────────────────         │
│  ● M3U playlist          Works everywhere, relative paths               │
│  ○ CUE sheet             Frame-accurate cue points (75 fps)             │
│  ○ Rekordbox XML         Pioneer CDJs and Rekordbox software  (coming)  │
│  ○ Traktor NML           Native Instruments Traktor           (coming)  │
│  ○ Serato crates         Serato DJ Pro                        (coming)  │
│  ○ Mixxx crate           Use ↺ Sync below for bidirectional   (coming)  │
│                                                                          │
│  Playlists to export                                                     │
│  ──────────────────────────────────────────────────────────────         │
│  ● All clusters (8 playlists, ~1,400 tracks)                            │
│  ○ Selected only:  [Deep Techno ▾]                                      │
│                                                                          │
│  Destination folder                                                      │
│  ──────────────────────────────────────────────────────────────         │
│  ~/Music/Playlists/                                        [Browse…]    │
│                                                                          │
│  ──────────────────────────────────────────────────────────────         │
│  [Export]                               [↺ Sync with Mixxx]             │
│                                                                          │
│  ✓ 8 playlists exported to ~/Music/Playlists/  (today 14:23)            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

M3U and CUE are implemented in core today. Rekordbox XML, Traktor, Serato, and Mixxx crate formats are shown but disabled pending #78 and #81. Mixxx Sync button is greyed if no DB path is configured in Preferences.

---

## Implementation order

```
#112  Navigation sidebar          ← architectural foundation, do first
  ├── #113  Library view
  │     └── #114  Preview panel   ← GStreamer player + cover art
  ├── #115  Playlists view
  ├── #101  Set Builder            ← depends on #36 harmonic, #85 energy arc
  └── #116  Export view            ← mostly wiring existing core exporters
```
