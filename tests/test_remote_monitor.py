"""Exercise replay event handling with Node; no browser or npm dependency."""
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which('node') is None, reason='Node required for replay JavaScript checks')
def test_remote_replay_frame_fallback_metrics_and_legacy_logs():
    html = Path(__file__).resolve().parents[1] / 'embodify_mcp' / 'monitor_page.html'
    script = r'''
const fs = require('fs'), assert = require('assert');
const html = fs.readFileSync(process.argv[1], 'utf8');
const js = html.split('<script>')[1].split('</script>')[0];
new Function(js); // Check the entire page's script syntax as well as the exercised functions.
const lang = js.slice(js.indexOf('// ---------- language'), js.indexOf('// ---------- helpers'));
const core = js.slice(js.indexOf('function newRun('), js.indexOf('// ---------- network'));
const urls = js.slice(js.indexOf('function frameURL('), js.indexOf('function preload('));
const render = js.slice(js.indexOf('function renderCall('), js.indexOf('let envKey ='));
const box = {innerHTML: ''};
const $ = () => box, S = {picked: null};
const dur = x => x.toFixed(3), esc = x => String(x), colorOf = () => '', argText = () => '';
const kvRows = x => JSON.stringify(x);
eval(lang + core + urls + render + `
const run = newRun({id:'demo'});
ingest(run, {kind:'scene', session:'test'});
ingest(run, {kind:'observation', label:'reset', step:0, t:0}); // Old logs omit image_cameras.
ingest(run, {kind:'observation', label:'m_0001', step:1, t:1, image_cameras:['cam']});
ingest(run, {kind:'observation', label:'m_0002', step:2, t:2, image_cameras:[], state:{x:2}});
ingest(run, {kind:'observation', label:'m_0003', step:3, t:3, image_cameras:[]});
ingest(run, {kind:'observation', label:'m_0004', step:4, t:4, image_cameras:['cam']});
assert.strictEqual(frameURL(run, run.frames[0], 'cam'), '/api/run/demo/frame/reset/cam');
assert.strictEqual(frameURL(run, run.frames[3], 'cam'), '/api/run/demo/frame/m_0001/cam');
assert.strictEqual(run.frames[2].state.x, 2);
assert.strictEqual(frameURL(run, run.frames[4], 'cam'), '/api/run/demo/frame/m_0004/cam');
ingest(run, {kind:'tool_call', name:'move_relative', ok:true, duration_s:1.5, transport:[
  {roundtrip_s:1.25, received_bytes:1024, sent_bytes:1024}], result:{}});
renderCall(run, run.frames[2]);
assert(box.innerHTML.includes('Backend round trip (incl. execution)'));
assert(box.innerHTML.includes('1.250'));
assert(box.innerHTML.includes('2.0 KiB'));
LANG = 'zh';
renderCall(run, run.frames[2]);
assert(box.innerHTML.includes('后端往返（含执行）'));
`);
'''
    result = subprocess.run([shutil.which('node'), '-e', script, str(html)], capture_output=True,
                            encoding='utf-8', timeout=60)
    assert result.returncode == 0, result.stderr
