# Flatpak Packaging Guide

This guide covers building and distributing Playchitect as a Flatpak.

## Distribution approach

Playchitect is distributed as a **self-hosted Flatpak bundle** — users download and
install a `.flatpak` file rather than adding a remote. This sidesteps Flathub's review
process (including its AI-code policy) while still providing the sandboxed Flatpak format.

A Flathub submission is tracked as a separate manual task in issue #60.

---

## Local build (development)

### Prerequisites

```bash
# Install flatpak-builder (Fedora)
sudo dnf install flatpak-builder

# Add Flathub remote (if not already present)
flatpak remote-add --user --if-not-exists flathub \
  https://dl.flathub.org/repo/flathub.flatpakrepo

# Install the GNOME Platform runtime (49 is current as of 2026-02)
flatpak install --user flathub org.gnome.Platform//49 org.gnome.Sdk//49
```

### Build and install locally

```bash
# From the repo root:
flatpak-builder \
  --user \
  --install \
  --force-clean \
  --repo=flatpak-repo \
  build-dir \
  packaging/flatpak/com.github.jameswestwood.Playchitect.yml

# Launch
flatpak run com.github.jameswestwood.Playchitect
```

### Export as a single-file bundle

```bash
flatpak build-bundle flatpak-repo \
  playchitect.flatpak \
  com.github.jameswestwood.Playchitect
```

---

## Installing from a bundle (end users)

Once a `.flatpak` bundle has been built (either locally or downloaded from a GitHub Release):

```bash
# One-time installation
flatpak install playchitect.flatpak

# Launch
flatpak run com.github.jameswestwood.Playchitect
```

---

## Automated CI build

The `.github/workflows/flatpak.yml` workflow builds the bundle automatically:

- **Trigger**: every GitHub Release, or manually via `workflow_dispatch`
- **Artifact**: `playchitect.flatpak` is attached to the release as a downloadable asset
- **Platform**: Ubuntu runner with `flatpak-builder` and GNOME Platform 49 installed

---

## Upgrading to offline builds (Flathub requirement)

The current manifest uses `pip install` during the build, which requires network access.
Flathub's build sandbox has no network. To produce an offline-capable manifest:

1. Install `flatpak-pip-generator`:
   ```bash
   pip install flatpak-pip-generator
   ```

2. Run the helper script:
   ```bash
   uv run python scripts/generate_flatpak_sources.py
   ```
   This writes `packaging/flatpak/python-deps.json` with explicit tarball URLs and
   SHA256 hashes for all runtime dependencies.

3. Update the manifest to use the generated module:
   ```yaml
   modules:
     - python-deps.json  # replaces the pip install module
     - name: playchitect
       ...
   ```

4. Remove `build-options: env: PIP_DISABLE_PIP_VERSION_CHECK` from the manifest.

5. Test the offline build:
   ```bash
   flatpak-builder --user --install --force-clean --disable-download \
     build-dir packaging/flatpak/com.github.jameswestwood.Playchitect.yml
   ```

---

## Manifest overview

`packaging/flatpak/com.github.jameswestwood.Playchitect.yml` contains:

| Field | Value |
|---|---|
| App ID | `com.github.jameswestwood.Playchitect` |
| Runtime | `org.gnome.Platform//49` |
| SDK | `org.gnome.Sdk//49` |
| Command | `playchitect-gui` |
| Filesystem | `xdg-music:ro`, `xdg-documents` (read-write for playlist output) |

### Sandbox permissions

| Permission | Reason |
|---|---|
| `--socket=wayland` | Primary display |
| `--socket=fallback-x11` | X11 fallback |
| `--share=ipc` | Shared memory (required with X11) |
| `--device=dri` | GPU acceleration |
| `--filesystem=xdg-music:ro` | Read music library |
| `--filesystem=xdg-documents` | Write playlist/CUE files |
| `--filesystem=home:ro` | Open arbitrary audio files via CLI/file manager |
| `--talk-name=org.freedesktop.portal.*` | File chooser dialog |

---

*Last updated: 2026-02-21*
