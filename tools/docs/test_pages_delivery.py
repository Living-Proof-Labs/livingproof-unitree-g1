# SPDX-License-Identifier: Apache-2.0
"""Exercise native artifact readback and the unchanged Pages privilege boundary."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import tarfile
import shutil
import unittest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = next(p for p in [ROOT/'.github/workflows/docs.yml', ROOT/'.github/workflows/manual.yml'] if p.exists())
DOC = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
BUILD = DOC['jobs']['build']
DEPLOY = DOC['jobs']['deploy']


def step(name):
    return next(s for s in DEPLOY['steps'] if s.get('name') == name)


def run(script, directory, **env):
    return subprocess.run(['bash', '-euo', 'pipefail', '-c', script], cwd=directory,
                          env=dict(os.environ, **env), capture_output=True, text=True)


class PagesDeliveryTests(unittest.TestCase):
    def test_replay_ref_cannot_select_checkout_or_executable(self):
        checkouts=[s for s in BUILD['steps'] if 'checkout@' in s.get('uses','')]
        self.assertEqual(len(checkouts),1)
        checkout=checkouts[0]
        expression=checkout['with']['ref']
        self.assertEqual(expression,'${{ github.event.pull_request.head.sha || github.sha }}')
        self.assertEqual(checkout['with']['fetch-depth'],'0')
        self.assertEqual(checkout['with']['fetch-tags'],'true')
        self.assertEqual(checkout['with']['persist-credentials'],'false')
        terms=expression.removeprefix('${{').removesuffix('}}').strip().split(' || ')
        for replay in ['refs/heads/untrusted','v999.0.0','b'*40,'$(touch injected)','--upload-pack=evil']:
            for pr in ['', 'c'*40]:
                with self.subTest(replay=replay,pr=pr):
                    context={'inputs.release_tag':replay,'github.sha':'a'*40,
                             'github.event.pull_request.head.sha':pr}
                    selected=next(context[term] for term in terms if context[term])
                    self.assertEqual(selected,pr or 'a'*40)
        identity=next(s for s in BUILD['steps'] if s.get('id')=='identity')
        self.assertGreater(BUILD['steps'].index(identity),BUILD['steps'].index(checkout))
        self.assertTrue(identity['run'].startswith('python tools/docs/release_identity.py identity Living-Proof-Labs/livingproof-unitree-'))
        for s in BUILD['steps']:
            # Operator inputs are data in env, never expressions choosing executed code.
            self.assertNotIn('inputs.',s.get('run',''))
            self.assertNotIn('inputs.',s.get('uses',''))

    def test_privileges_tokens_and_final_lock_boundary(self):
        self.assertEqual(DOC['permissions'], {'contents':'read'})
        self.assertNotIn('permissions', BUILD)
        self.assertEqual(DEPLOY['permissions'], {'contents':'read','pages':'write','id-token':'write'})
        model = 'go1' if WORKFLOW.stem == 'docs' else 'g1'
        self.assertEqual(DEPLOY['concurrency'], {'group':model+'-pages','cancel-in-progress':'false'})
        self.assertNotIn('GH_TOKEN', BUILD.get('env', {}))
        self.assertNotIn('GH_TOKEN', DEPLOY.get('env', {}))
        for s in DEPLOY['steps']:
            self.assertNotIn('checkout@', s.get('uses',''))
            self.assertNotIn('tools/', s.get('run',''))
        token_steps=[s['name'] for s in BUILD['steps'] if 'GH_TOKEN' in s.get('env',{})]
        self.assertEqual(token_steps,['Verify published release and exact source identity'])
        token=next(s for s in BUILD['steps'] if s.get('name')==token_steps[0])['env']['GH_TOKEN']
        self.assertIn("github.event_name != 'pull_request'",token)
        self.assertIn("inputs.release_tag != ''",token)
        self.assertIn("github.ref == 'refs/heads/main'",token)
        self.assertEqual([s['name'] for s in DEPLOY['steps'][-3:]],[
            'Recheck immutable published release inside deployment lock',
            'Require current main inside deployment lock','Deploy verified manual'])
        self.assertIn("needs.release.outputs.published == 'true'", (ROOT/'.github/workflows/release.yml').read_text())

    def test_bounded_same_run_id_and_name_routes(self):
        upload=next(s for s in BUILD['steps'] if s.get('id')=='pages-artifact')
        self.assertEqual(upload['with']['name'],'github-pages-${{ github.run_attempt }}')
        self.assertEqual(DEPLOY['steps'][-1]['with']['artifact_name'],upload['with']['name'])
        self.assertEqual(BUILD['outputs']['pages-id'],'${{ steps.pages-artifact.outputs.artifact-id }}')
        self.assertEqual(BUILD['outputs']['pages-digest'],'${{ steps.pages-package.outputs.tar-sha256 }}')
        self.assertEqual(len([s for s in DEPLOY['steps'] if 'download-artifact@' in s.get('uses','')]),4)
        for route,selector,value in [('id','artifact-ids','${{ needs.build.outputs.pages-id }}'),('name','name',upload['with']['name'])]:
            first=next(s for s in DEPLOY['steps'] if s.get('id')==f'pages-{route}-first')
            retry=next(s for s in DEPLOY['steps'] if s.get('id')==f'pages-{route}-retry')
            self.assertEqual(first['continue-on-error'],'true')
            self.assertEqual(retry['if'],f"steps.pages-{route}-first.outcome == 'failure'")
            self.assertNotIn('continue-on-error',retry)
            self.assertEqual(first['uses'],retry['uses'])
            for s in [first,retry]:
                self.assertEqual(set(s['with']),{selector,'path'})
                self.assertEqual(s['with'][selector],value)
            self.assertNotEqual(first['with']['path'],retry['with']['path'])
        verify=step('Require matching ID and name readback of the exact tar')
        self.assertEqual(verify['env']['EXPECTED_TAR'],'${{ needs.build.outputs.pages-digest }}')
        self.assertEqual(len([s for s in DEPLOY['steps'] if s.get('run')=='sleep 4']),2)

    def test_readback_rejects_missing_wrong_bytes_or_prior_attempt(self):
        for case in ['exact','missing','id-tampered','name-tampered','prior-attempt','empty-path','bad-digest']:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); by_id=root/'id'; named=root/'name'
                location=by_id/('github-pages-1' if case=='prior-attempt' else 'github-pages-2')
                location.mkdir(parents=True);named.mkdir()
                (location/'artifact.tar').write_bytes(b'changed' if case=='id-tampered' else b'sealed')
                if case!='missing':(named/'artifact.tar').write_bytes(b'changed' if case=='name-tampered' else b'sealed')
                result=run(step('Require matching ID and name readback of the exact tar')['run'],root,
                    EXPECTED_NAME='github-pages-2',EXPECTED_TAR='bad' if case=='bad-digest' else hashlib.sha256(b'sealed').hexdigest(),
                    ID_READBACK='' if case=='empty-path' else str(by_id),NAME_READBACK=str(named))
                self.assertEqual(result.returncode==0,case=='exact',result.stderr)

    def test_packaging_preserves_only_sealed_bytes(self):
        package=next(s for s in BUILD['steps'] if s.get('id')=='pages-package')
        for case in ['exact','changed','added','symlink']:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); folder='docs' if WORKFLOW.stem=='docs' else 'manual'
                html=root/f'artifacts/{folder}/html';html.mkdir(parents=True)
                helper=root/'tools/docs/release_identity.py';helper.parent.mkdir(parents=True)
                shutil.copyfile(ROOT/'tools/docs/release_identity.py',helper)
                record={'run_id':None,'run_attempt':None}
                (html/'build.json').write_text(json.dumps(record))
                (html/'index.html').write_text('tested')
                (html/'.nojekyll').write_text('')
                files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in html.iterdir()}
                (html.parent/'seal.json').write_text(json.dumps({'identity':record,'files':files}))
                if case=='changed':(html/'index.html').write_text('different')
                if case=='added':(html/'extra.html').write_text('extra')
                if case=='symlink':(html/'link').symlink_to(html/'index.html')
                output=root/'output'
                result=run(package['run'],root,RUNNER_TEMP=tmp,GITHUB_OUTPUT=str(output),GITHUB_RUN_ID='',GITHUB_RUN_ATTEMPT='')
                self.assertEqual(result.returncode==0,case=='exact',result.stderr)
                if case=='exact':
                    archive=root/'artifact.tar'
                    self.assertEqual(output.read_text().strip(),'tar-sha256='+hashlib.sha256(archive.read_bytes()).hexdigest())
                    with tarfile.open(archive) as tar:
                        self.assertEqual(sorted(m.name for m in tar if m.isfile()),['./.nojekyll','./build.json','./index.html'])
                        self.assertEqual(tar.extractfile('./index.html').read(),b'tested')

    def test_uploaded_id_and_stale_attempt_rejected(self):
        for value in ['','0','-1','abc','1; true','42']:
            with self.subTest(value=value),tempfile.TemporaryDirectory() as tmp:
                result=run(step('Require a valid Pages artifact ID')['run'],tmp,UPLOADED_ID=value)
                self.assertEqual(result.returncode==0,value=='42')
        for attempt in ['1','2','']:
            with tempfile.TemporaryDirectory() as tmp:
                result=run(step('Reject retained build from an earlier attempt')['run'],tmp,BUILD_ATTEMPT=attempt,GITHUB_RUN_ATTEMPT='2')
                self.assertEqual(result.returncode==0,attempt=='2')

    def test_inline_release_read_requires_immutable_exact_release(self):
        for case in ['exact','annotated','mutable','missing-immutable','draft','prerelease','unpublished','wrong-tag','wrong-commit','api-failure','wrong-version']:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                gh=root/'gh'
                gh.write_text('''#!/usr/bin/env python3
import json,os,sys
case=os.environ['CASE']; endpoint=sys.argv[2]
if case=='api-failure':sys.exit(1)
if '/releases/tags/' in endpoint:
 r=dict(tag_name='v1.2.3',draft=False,prerelease=False,immutable=True,published_at='2026-11-22')
 if case=='mutable':r['immutable']=False
 if case=='missing-immutable':del r['immutable']
 if case in ('draft','prerelease'):r[case]=True
 if case=='unpublished':r['published_at']=None
 if case=='wrong-tag':r['tag_name']='v1.2.4'
 print(json.dumps(r))
else:
 print(('tag' if case=='annotated' and '/git/ref/' in endpoint else 'commit')+'\\t'+('b' if case=='wrong-commit' else 'a')*40)
''')
                gh.chmod(0o755)
                result=run(step('Recheck immutable published release inside deployment lock')['run'],root,
                    PATH=str(root)+':'+os.environ['PATH'],CASE=case,RELEASE_TAG='v1.2.3',RELEASE_VERSION='1.2.4' if case=='wrong-version' else '1.2.3',
                    GITHUB_EVENT_NAME='workflow_run',GITHUB_REPOSITORY='owner/repo',GITHUB_SHA='a'*40)
                self.assertEqual(result.returncode==0,case in ['exact','annotated'],result.stderr)

    def test_final_main_check_rejects_advanced_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            gh=Path(tmp)/'gh';gh.write_text('#!/bin/sh\nprintf "%s\\n" "$REMOTE_MAIN"\n');gh.chmod(0o755)
            for sha in ['a'*40,'b'*40,'']:
                result=run(step('Require current main inside deployment lock')['run'],tmp,
                    PATH=tmp+':'+os.environ['PATH'],REMOTE_MAIN=sha,CANDIDATE='a'*40)
                self.assertEqual(result.returncode==0,sha=='a'*40)


if __name__=='__main__':unittest.main()
