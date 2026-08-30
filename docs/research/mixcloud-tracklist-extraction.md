# Mixcloud tracklist extraction (verified)

**Verified:** 2026-08-30, against the four BEORMAN set pages (see `data/own_sets.yaml`).

## Finding

The full ordered tracklist is embedded in every public set page's HTML as React/Relay
flight data, even though the UI gates full display behind Mixcloud Premium:

```
canShowTracklist: !1            # UI gate only — the data ships regardless
sections: [{__typename:"TrackSection", artistName:"Abdulla Rashim", songName:"Crossing Qalandiya"}, ...]
```

Extraction = one plain GET of the set page, then parse `TrackSection` entries
(`artistName`, `songName`) in payload order. No auth, cookies, CSRF, or GraphQL
session needed. The page payload carries **no timestamps** — ordering only.

Evidence: `beorman-heave-wave-mix-26` page yielded 14 TrackSection entries opening
with Abdulla Rashim — "Crossing Qalandiya", matching the 2026-06-26 CUE sheet.

## Caveats

- Tracklist exists only when the uploader provided one (same limit the browser
  extension documents). Underground sets full of `ID - ID` will still have gaps.
- The flight format ($R[n] reference indirection) is brittle to upstream Mixcloud
  changes. Mitigation: every fetched page is cached to disk before parsing
  (`playchitect/scrape/client.py` discipline), so parsing is repeatable offline.
- Politeness: minimum 5s between live requests, descriptive personal-research UA.

## Reference

`trepDev/mixcloud-with-tracklist` (MPL-2.0) — browser extension that surfaces these
tracklists via the authenticated GraphQL route from a browser session. Not needed
for our harvest (page-embedded data suffices); kept as a fallback reference if
Mixcloud ever stops embedding the payload.

## Where this is used

- **TASK-23** (golden eval harvest): primary source for `hard-minimal-industrial-techno-mix`
  (recorded Oct 2025, no CUE exists); cross-check source for the three CUE-backed sets
  (CUE preferred on conflict — it is frame-accurate).
- **Enhancement track (TASK-21/TASK-H1):** Mixcloud is a viable additional source for
  other artists' sets via this method. Not in default scope (1001tracklists + MixesDB
  only); decision deferred to the TASK-H1 human gate.
