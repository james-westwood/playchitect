# Playchitect — Release Strategy

## Target Audience

Primary: Fedora / GNOME desktop users running a Linux audio workstation.
Secondary: Any Linux user comfortable with a terminal.

---

## Distribution Channels

### Priority order

| Priority | Channel | Status | Notes |
|---|---|---|---|
| 1 | **PyPI** (#17) | 🚧 Planned | CLI immediately accessible everywhere; no AI policy |
| 2 | **COPR** | 🚧 Planned | Fedora-native; DNF integration; no AI policy |
| 3 | **Self-hosted Flatpak** | 🔵 Optional | Keeps Flatpak format without Flathub gatekeeping |
| 4 | **AUR** | 🔵 Optional | Low effort; reaches Arch / Manjaro GNOME users |
| 5 | **Flathub** (#16) | ⚠️ Risk | AI policy may block submission — see below |

---

## Channel Details

### 1. PyPI (issue #17)

`pip install playchitect` / `uv tool install playchitect`

- `pyproject.toml` is already well-formed
- Entry points already defined: `playchitect`, `playchitect-gui`, `playchitect-install-desktop`
- GTK4 / PyGObject cannot ship via PyPI (system dependency) — document this clearly
- Publish with `uv build && uv publish`
- Version: tag `v1.0.0` on `main`, then `uv publish --token $PYPI_TOKEN`

**Effort**: Low — `pyproject.toml` is ready

---

### 2. COPR (Fedora Community Package Repo)

Makes the app available via `dnf install`:

```bash
dnf copr enable jameswestwood/playchitect
dnf install playchitect
```

**Steps**:
1. Create a FAS account at https://accounts.fedoraproject.org
2. Create a COPR project at https://copr.fedorainfracloud.org
3. Write an RPM `.spec` file — `pyp2rpm` can auto-generate a first draft from the PyPI package
4. Build the SRPM and submit to COPR
5. COPR builds for multiple Fedora releases automatically

**Effort**: Medium — RPM spec has a learning curve but `pyp2rpm` reduces it significantly for pure-Python packages

**No AI policy.**

---

### 3. Self-hosted Flatpak Remote

Build the Flatpak locally and host the repo on GitHub Pages. Users add the remote once:

```bash
flatpak remote-add playchitect https://james-westwood.github.io/playchitect-flatpak/
flatpak install playchitect com.github.jameswestwood.Playchitect
```

The `.desktop`, AppStream XML, and icons from Milestone 4 translate directly into the Flatpak.

**Steps**:
1. Write `packaging/flatpak/com.github.jameswestwood.Playchitect.yml`
2. Build with `flatpak-builder`
3. Export the repo to a static directory
4. Host on GitHub Pages via a CI job

**Effort**: Medium — manifest writing + CI publishing setup

**No AI policy.**

---

### 4. AUR (Arch User Repository)

A `PKGBUILD` file (~30 lines for a pure-Python app) submitted to the AUR.

```bash
yay -S playchitect   # or paru, or manual makepkg
```

**Effort**: Low

**No AI policy.**

---

### 5. Flathub (issue #16)

Flathub is the largest Linux app store and would give the most discoverability. However:

#### ⚠️ AI Policy

Flathub's [submission requirements](https://docs.flathub.org/docs/for-app-authors/requirements) state:

> "Submissions or changes where most of the code is written by or using AI without any
> meaningful human input, review, justification or moderation of the code are not allowed."
>
> "Submission pull requests must not be generated, opened, or automated using AI tools or agents."

**Rule 1** (submission PR opened by AI) is easy to comply with — James must open the PR manually.

**Rule 2** (most code AI-written without meaningful human input) is the grey area. Playchitect
has been substantially AI-assisted, but James has:
- Provided all requirements and architectural direction
- Reviewed and approved every PR
- Made all key decisions (algorithms, library choices, UX)

Flathub reviews case-by-case. The code quality and GNOME HIG compliance will matter more to
reviewers than the development method. Attempting submission is worthwhile — worst case is a
rejection with feedback.

**If rejected**: fall back to self-hosted Flatpak remote (option 3) and COPR (option 2), which
reach the same Fedora/GNOME audience without gatekeeping.

---

## Version Strategy

- Tag `v1.0.0` on `main` once Milestone 6 is complete
- Use semantic versioning: `MAJOR.MINOR.PATCH`
- Changelog in `CHANGELOG.md` (to be created at release time)
- GitHub Release with attached `.whl` and source tarball

---

## Release Checklist

### Pre-release
- [ ] All Milestone 6 issues closed (#16, #17)
- [ ] `version` in `pyproject.toml` set to `1.0.0`
- [ ] `CHANGELOG.md` written
- [ ] README screenshots updated
- [ ] `uv run pytest -q` passes on a clean venv
- [ ] `uv run pre-commit run --all-files` passes

### PyPI
- [ ] `uv build` produces clean `.whl` and `.tar.gz`
- [ ] Test install in a fresh venv: `pip install dist/playchitect-1.0.0-py3-none-any.whl`
- [ ] `uv publish` to TestPyPI first, then PyPI

### COPR
- [ ] RPM `.spec` file written and tested with `rpmbuild`
- [ ] SRPM submitted to COPR project
- [ ] Test `dnf install` on a clean Fedora VM

### Flathub (attempt)
- [ ] `packaging/flatpak/com.github.jameswestwood.Playchitect.yml` written
- [ ] `flatpak-builder` build succeeds locally
- [ ] `appstream-util validate` passes
- [ ] `desktop-file-validate` passes
- [ ] Submission PR opened **manually by James** (not Claude/Gemini)
- [ ] No AI review tools requested in the Flathub PR

### GitHub Release
- [ ] Tag `v1.0.0`: `git tag -a v1.0.0 -m "Release v1.0.0"`
- [ ] Push tag: `git push origin v1.0.0`
- [ ] Create GitHub Release with release notes

---

*Last updated: 2026-02-21*
