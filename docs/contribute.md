# Contribute, help and licensing

Open a focused pull request in the [source repository](https://github.com/Living-Proof-Labs/livingproof-unitree-g1).
Use [issues](https://github.com/Living-Proof-Labs/livingproof-unitree-g1/issues)
for ordinary questions and reproducible bugs. No dedicated security reporting
route was found during this adoption; do not post sensitive reports publicly.

This docs build follows the [documentation recipe at v7.10.8](https://github.com/Living-Proof-Labs/collaboration-guidelines/blob/v7.10.8/docs/documentation-recipe.md)
and pins that release's brand by commit and digest. Existing product/release
workflow pins remain unchanged. Current manual, organization home and source
navigation links are bounded exceptions to cross-repository version pinning.

Repository code and prose retain the existing [Apache-2.0 license](https://github.com/Living-Proof-Labs/livingproof-unitree-g1/blob/main/LICENSE).
Downloaded model/controller licenses are listed in the [guide](guide.md#files);
they retain their upstream scopes. The shared brand retains its upstream license.

The README remains canonical for the robot guide; the build stages it without
maintaining a second editable copy. Edit `docs/` for the manual's other pages.

## Cache freshness and upgrades

The shared brand stylesheet is verified against `docs-brand.lock.json` and
referenced as `assets/brand.css?v=<sha256>`. Changed bytes get a new URL; unchanged
bytes retain their URL. Every generated page is checked for the correct
stylesheet and digest. `build.json` records the source commit and brand identity.
The file path stays stable, so cached old HTML can still fetch a stylesheet
after an update instead of requesting a deleted asset. This is a cache-busting
URL, not an immutable historical resource: an uncached old version URL can
serve the current CSS. Old HTML therefore does not promise a historical snapshot.

HTML freshness is separate. GitHub Pages currently serves this manual's HTML
and build metadata with `Cache-Control: max-age=600`, `ETag` and `Last-Modified`
(observed 14 November 2026). This repository cannot set custom response headers
through a Pages `_headers` file. The operational target is a normal navigation
or reload receiving the new build within ten minutes after the new deployment
is served. Verify actual response headers after deployment; a hosting change
may require a different freshness target. Do not claim immediate updates based
only on versioned CSS or a green build.

An already-open tab, including one restored by Back/Forward, may keep its old
HTML and styles until a normal reload. That is expected; this manual does not
poll or force readers to reload. No service worker or progressive-web-app
manifest is used for cache invalidation. If an old page persists after the
freshness window and normal reload, capture its URL, Version and source page,
response headers and build identity before clearing browser data.

Run `python -m unittest discover -s tools/docs/tests` for the warm asset-cache
key regression test, then `python tools/docs/build.py` for rendered-reference
and digest validation. These tests do not reproduce a particular browser's
cache or history. For deployment acceptance, warm a browser on build N, deploy
N+1, navigate/reload after the freshness window, and verify HTML source identity,
stylesheet URL, `build.json`, dark/light toggles and Back/Forward restoration.
Record results and any explicit reload required; never call an untested browser
path verified. The original laptop's prolonged stale view was not reproduced.
