# Research: Mixxx Database Integration (Read-only Sync)

This document outlines the research for implementing read-only synchronization between Playchitect and the Mixxx DJ software library.

## Database Location
Mixxx stores its library in a single SQLite database file: `mixxxdb.sqlite`.

Default locations:
- **Linux**: `~/.mixxx/mixxxdb.sqlite`
- **macOS**: `~/Library/Application Support/Mixxx/mixxxdb.sqlite`
- **Windows**: `%LOCALAPPDATA%\Mixxx\mixxxdb.sqlite`

## Schema XML
The official source of truth for the schema is maintained in the Mixxx source code: `mixxx/res/schema.xml`.

## Key Tables and Columns

### 1. `track_locations`
Used to map tracks to physical files.
- `location`: Absolute file path (Primary Key for matching).
- `fs_deleted`: 1 if the file is missing/deleted from disk.

### 2. `library`
Contains metadata and performance stats.
- `rating`: Track rating (0-5 stars).
- `timesplayed`: Total play count.
- `last_played_at`: ISO8601 or Unix timestamp of last performance.
- `samplerate`: Sample rate of the audio (critical for cue conversion).

### 3. `cues`
Stores saved cue points and loops.
- `position`: The offset **in samples**.
  - *Conversion*: `seconds = position / samplerate`
- `type`: Type of cue (Hotcue, Loop, etc).
- `label`: User-defined name for the cue.

## Implementation Strategy

### Connection Security
To avoid locking Mixxx or corrupting the database, the connection must be opened in **Read-Only** mode:
```python
# URI syntax for read-only access
db_uri = f"file:{db_path}?mode=ro"
connection = sqlite3.connect(db_uri, uri=True)
```

### Matching Logic
1. Scan local directories in Playchitect.
2. For each track, check if the absolute path exists in the Mixxx `track_locations` table.
3. If matched, join with the `library` table to enrich the `TrackMetadata` object with `rating`, `playcount`, and `last_played`.

### Cue Enrichment
When displaying a track in the GUI, Playchitect can pull associated cues from the Mixxx `cues` table to provide better visual markers for sequencing.

## Sample Query
```sql
SELECT
    tl.location,
    l.rating,
    l.timesplayed,
    l.last_played_at,
    l.samplerate
FROM library l
JOIN track_locations tl ON l.location = tl.id
WHERE tl.fs_deleted = 0;
```
