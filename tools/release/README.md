# One-off v2.1.4 recovery

Tracks issue #46 and the partial publication after PR #44. This is not a general
release publisher. The tag already exists at `b5cfcb0d0268015aa5f9ce0914439c7ade67269d`.
Do not delete or move it, or blindly rerun semantic-release: it sees the existing
tag and may report no new release while the GitHub release remains missing.

`recover_v214.py` is fixed to this repository, tag, commit and previous v2.1.3
commit. The checked-in notes were generated from those two tags using
collaboration-guidelines v9.5.4's release.config.mjs, semantic-release 25.0.9,
release-notes-generator 14.1.1 and conventionalcommits 8.0.0. A fixed SHA-256 binds
the reviewed notes. The semantic-release default-channel note is
`{"channels":[null]}`. Existing matching state is left alone; mismatches stop.
The script creates no tags, changes no ref and makes no PATCH/PUT/DELETE calls.
A normal GitHub release is created against the existing tag, with make_latest
false so a newer release cannot be displaced by this historical repair.

After review, merge and green exact-head CI:
1. Dispatch **Recover existing v2.1.4 publication** on main with apply **false**.
   Review the fixed tag/SHA and missing-state output. No write occurs.
2. Recheck that approval covers this precise repair, then dispatch on main with
   apply **true**. The write job uses the ordinary workflow token with only
   contents:write and the same concurrency group as normal semantic releases.
3. Retain the run and final tag/note/release readback on #46 and #44. Keep them
   unfinished if any final check fails. This is pipeline, not robot evidence.

Any unexpected existing object, HTTP error other than an optional read's 404,
or uncertain write stops the run. Do not auto-retry a POST. Inspect state first;
a subsequent run can safely resume a partially completed repair. Orphan Git
objects from a stopped pre-ref creation are harmless and are not publications.
Remove the one-off workflow in a later reviewed cleanup after recovery is proven.

Offline tests: `python -m pytest tests/test_release_recovery.py`.

Concurrency limitation: the shared lock excludes normal release jobs, not an
external administrator deleting a tag. GitHub's release-create API can create a
missing tag; therefore the tag read and release POST are not an atomic
create-only-if-tag-still-exists operation. Final readback requires the exact
original target. Do not dispatch while external tag mutation is planned; stop
if it is observed. No tag protections or other settings are changed here.
