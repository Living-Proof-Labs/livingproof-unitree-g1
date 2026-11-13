# Verification and publishing

From the checkout, with Python 3.12, uv and Task 3.54.0 installed:

```sh
uv venv .venv-docs
uv pip sync --python .venv-docs/bin/python tools/docs/requirements.lock
.venv-docs/bin/python tools/docs/build.py
```

Open `artifacts/manual/html/index.html` through a local HTTP server. The build
checks local page/fragment destinations and shared branding. It generates Task
reference using native `task --list --json` and `task --summary`; it does not run
robot tasks. The environment needs network access for the pinned brand stylesheet.

The Repository manual workflow builds pull requests, retains the rendered manual
for 14 days, and deploys only main through GitHub Pages. A maintainer must configure
Pages to use GitHub Actions and permit deployment from main to `github-pages`.
Changing repository/security settings requires separate confirmation. This PR
neither changes settings nor proves that a site is live.

Expected publication: <https://docs.livingprooflabs.com/livingproof-unitree-g1/>.
Verify the actual deployment URL after merge, including README navigation, nested
pages, keyboard/mobile use, search, and the source SHA in `build.json`.

## Product evidence and adoption limits

[Verify workflow runs](https://github.com/Living-Proof-Labs/livingproof-unitree-g1/actions/workflows/verify.yml)
retain simulation evidence, assessment and reports described in the guide. This
manual does not claim a passing robot grade or measured coverage, and does not
copy stale results. Inspect a run and its exact revision for actual outcomes.

This is a basic manual adoption: generated native CLI help, public API reference,
and embedded scenario results remain follow-up work. The guide preserves existing
commands and the source owns the robot/configuration contracts; specs owns schemas.
No runtime changes or runtime verification are part of this docs-only change.
Independent review and live browser/deployment verification remain release checks.
