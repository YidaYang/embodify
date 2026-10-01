"""CPU-only contract doubles; these do not claim Isaac/PhysX validation."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
from embodify_mcp.backend.base import McpToolError
from embodify_mcp.backend.wire import pack_snapshot, unpack_snapshot
from embodify_mcp.mcp import McpSession
from embodify_mcp.robodojo.config import ARMS, CAMERAS, Settings
from embodify_mcp.robodojo.episode import EpisodeBackend, decode_observation, target_action
from embodify_mcp.robodojo.tasks import TASKS

#: What an environment reports for a task whose instruction upstream generates per layout.
GENERATED = 'stack the blocks from bottom to top in the order of blue, green, and red, then reset the robot arm.'

class Driver:
    def __init__(self, *, final_success=False, immobile=False, task='stack_blocks', budget=None, instruction=None):
        self.task=task; self.budget=TASKS[task].step_limit if budget is None else budget
        self.ended=False; self.success=False; self.final_success=final_success
        self.immobile=immobile; self.actions=[]; self.closed=False; self.finishes=0
        if instruction is None: instruction=TASKS[task].instruction or GENERATED
        self.raw={'instruction':instruction,'state':{},'vision':{},'scene_layout':'PRIVATE','reward':1}
        for i,a in enumerate(ARMS):
            self.raw['state'][a+'_ee_pose']=[i*.5,0,1,1,0,0,0]
            self.raw['state'][a+'_ee_joint_state']=[1.]
        for c in CAMERAS:
            self.raw['vision'][c]={'color':np.full((16,16,4),[255,17,3,255],dtype=np.uint8)}
    def reset(self):return self.observe()
    def observe(self):return copy.deepcopy(self.raw)
    def step(self,action):
        self.actions.append(copy.deepcopy(action))
        if not self.immobile:
            for a in ARMS:self.raw['state'][a+'_ee_pose']=action[a+'_ee_pose']
        for a in ARMS:self.raw['state'][a+'_ee_joint_state']=action[a+'_ee_joint_state']
        return self.observe()
    def finish(self):
        self.finishes+=1; self.ended=True; self.success=self.final_success
        return {'robodojo_result':{'details':{'0':{'success':self.success,'score':float(self.success),'layout_id':0}}}}
    def close(self):self.closed=True

def start(driver=None):
    b=EpisodeBackend(driver or Driver());b.reset(b.catalogue.task,0);return b

def test_source_rgb_wxyz_and_unknown_gripper():
    d=Driver();d.raw['state']['left_ee_pose']=[1,2,3,0,1,0,0]
    s=decode_observation(d.raw,0,dict.fromkeys(ARMS,'open'))
    assert s.arms['left'].quat.tolist()==[1,0,0,0]
    assert s.arms['left'].gripper_opening is None
    assert s.images['cam_head'][0,0].tolist()==[255,17,3]
    r=unpack_snapshot(pack_snapshot(s,'rgb-zlib'))
    assert r.arms['left'].gripper_opening is None
    assert np.array_equal(r.images['cam_head'],s.images['cam_head'])

@pytest.mark.parametrize('bad',[0,float('nan')])
def test_reject_invalid_quaternion(bad):
    d=Driver();d.raw['state']['left_ee_pose'][3:]=[bad]*4
    with pytest.raises(ValueError):decode_observation(d.raw,0,dict.fromkeys(ARMS,'open'))

def test_action_is_full_two_arm_and_retains_other_latch():
    b=start();b.set_gripper('right',True,max_steps=1)
    before=b.observe()
    action=target_action(before,{'left':'open','right':'close'},'left',[.1,0,1],[0,0,0,1])
    assert action['right_ee_pose']==[.5,0,1,1,0,0,0]
    assert action['right_ee_joint_state']==[0.]
    assert action['left_ee_pose']==[.1,0,1,1,0,0,0]

def test_action_budget_and_callback_indices():
    b=start();seen=[]
    r=b.move_to('left',np.array([.2,0,1]),np.array([0,0,0,1]),max_steps=3,on_step=lambda s:seen.append(s.step))
    assert seen==[1,2,3] and r.executed_steps==3 and r.stop_reason=='limit'
    assert len(b.driver.actions)==3
    assert abs(r.snapshot.arms['left'].pos[0]-.06)<1e-9
    assert b.observe().step==3

def test_gripper_reports_command_not_settled_and_zero_budget_keeps_latch():
    b=start();r=b.set_gripper('left',True,max_steps=0)
    assert r.snapshot.arms['left'].gripper_command=='open' and not b.driver.actions
    r=b.set_gripper('left',True,max_steps=1)
    assert r.stop_reason=='command_applied' and r.executed_steps==1
    assert r.snapshot.arms['left'].gripper_opening is None

def test_stalled_and_terminal_do_not_spin():
    b=start(Driver(immobile=True))
    r=b.move_to('left',np.array([.2,0,1]),np.array([0,0,0,1]),max_steps=20)
    assert r.stop_reason=='stalled' and r.executed_steps==6
    b.driver.ended=True
    assert b.set_gripper('left',True,max_steps=10).executed_steps==0

def test_mcp_images_arm_schema_private_boundaries_and_final_evaluation(tmp_path):
    d=Driver(final_success=True);b=EpisodeBackend(d);s=McpSession(b,output_root=tmp_path)
    results=[]
    for tool,args in [('reset_task',{}),('observe',{}),('move_relative',{'arm':'left','delta_xyz':[.04,0,0]}),('set_gripper',{'arm':'right','state':'close'}),('stop_episode',{})]:
        result=s.call_tool(tool,args);assert not result.get('isError'),result
        results.append(result)
    text=json.dumps(results)
    assert all(word not in text for word in b.info.forbidden_terms)
    assert len([x for x in results[0]['content'] if x['type']=='image'])==3
    assert results[3]['structuredContent']['action']['opening_after_m'] is None
    summary=json.loads((s.journal.dir/'summary.json').read_text(encoding='utf-8'))
    assert summary['success_at_stop'] is True and summary['success'] is True
    assert summary['physics_audit']['robodojo_result']['details']['0']['score']==1
    assert d.finishes==1 and d.closed
    move=next(t for t in s.tools() if t['name']=='move_relative')
    assert 'arm' in move['inputSchema']['required']
    assert 'null' in move['description']

@pytest.mark.parametrize('args',[{'task_index':1},{'seed':1},{'suite':'libero_spatial'}])
def test_agent_cannot_select_private_configuration(args):
    with pytest.raises(McpToolError):EpisodeBackend(Driver()).catalogue.resolve(args,locked=False)

def test_settings_reject_invalid_values():
    for kw in [{'seed':True},{'layout_index':-1},{'image_size':0},{'startup_timeout_s':float('nan')}]:
        with pytest.raises(ValueError):Settings(source='x',python='p',output='o',**kw)

def test_supervisor_worker_over_actual_p2_pipe(tmp_path):
    from embodify_mcp.backend.robodojo import RoboDojoBackend
    config=tmp_path/'config.json';config.write_text(json.dumps({'source':'unused','python':sys.executable,'output':str(tmp_path)}))
    tests=Path(__file__).resolve().parent
    code=('import sys;sys.path.insert(0,'+repr(str(tests))+');'
          'from test_robodojo_backend import Driver;'
          'from embodify_mcp.robodojo.episode import EpisodeBackend;'
          'from embodify_mcp.backend.serve import serve_connection;'
          'from embodify_mcp.backend.transport import LineTransport;'
          'serve_connection(EpisodeBackend(Driver(final_success=True)),LineTransport(sys.stdin.buffer,sys.stdout.buffer,lambda:None))')
    b=RoboDojoBackend(config,worker_command=[sys.executable,'-u','-c',code])
    s=McpSession(b,output_root=tmp_path/'logs')
    try:
        for _ in range(2):
            assert not s.call_tool('reset_task').get('isError')
            assert not s.call_tool('set_gripper',{'arm':'left','state':'close'}).get('isError')
            assert not s.call_tool('stop_episode').get('isError')
            summary=json.loads((s.journal.dir/'summary.json').read_text(encoding='utf-8'))
            assert summary['success_at_stop'] is True
            assert b.worker is None
    finally:s.close()

@pytest.mark.parametrize('final_success',[False,True])
@pytest.mark.parametrize('episode_budget',[550,10000])
def test_native_finalizer_matches_pinned_official_writer(tmp_path,final_success,episode_budget):
    import importlib.util
    from types import SimpleNamespace
    from embodify_mcp.robodojo.isaac_driver import IsaacDriver
    spec=importlib.util.spec_from_file_location('official_writer',Path(__file__).parent/'fixtures/robodojo_run_eval.py')
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    class Env:
        run_eval=oracle.run_eval
        def run_reward(self):raise AssertionError('predicates must not be registered twice')
        def get_score(self):raise AssertionError('scores must not be registered twice')
        def eval_one_episode(self):raise AssertionError('actions must not be repeated')
        def get_running_env_idx_list(self):return [] if self.end_flag[0] else [0]
        def save_video(self,*args):pass
        def _abort_video_writers(self):pass
        def persist_resume_manifest(self):pass
    e=Env()
    e.end_flag=[False];e.success=[True];e.unstable_envs=set();e.unstable_nums=0
    e.reward_manager=SimpleNamespace(get_reward=lambda **kwargs:[float(final_success)],get_score=lambda:[37.])
    e.eval_batch=False;e.episode_nums=1;e.success_nums=e.fail_nums=0;e.total_score=0
    e.env_seeds=[3];e.save_dir=str(tmp_path);e.eval_result={'details':{}}
    d=IsaacDriver(Settings(source='unused',python=sys.executable,output=str(tmp_path),episode_budget=episode_budget));d.env=e
    audit=d.finish()
    assert audit['episode_budget']==episode_budget and audit['native_episode_budget']==550
    assert audit['budget_overridden'] is (episode_budget!=550)
    details=audit['robodojo_result']['details']['0']
    assert details=={'layout_id':3,'success':final_success,'score':1. if final_success else .37}
    assert d.success is final_success and d.finish() is audit
    assert e.get_running_env_idx_list()==[]
    with pytest.raises(AssertionError):e.run_reward()

def test_nested_p2_robodojo_disconnect_and_reset(tmp_path):
    # Nested actual byte streams: host MCP -> P2 service -> isolated pilot worker.
    from embodify_mcp.backend.remote import RemoteBackend
    from embodify_mcp.backend.robodojo import RoboDojoBackend
    import socket,threading
    from embodify_mcp.backend.serve import serve_connection
    from embodify_mcp.backend.transport import socket_transport
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps({'source':'x','python':sys.executable,'output':str(tmp_path)}))
    code=('import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');'
          'from test_robodojo_backend import Driver;'
          'from embodify_mcp.robodojo.episode import EpisodeBackend;'
          'from embodify_mcp.backend.serve import serve_connection;'
          'from embodify_mcp.backend.transport import LineTransport;'
          'serve_connection(EpisodeBackend(Driver(final_success=True)),LineTransport(sys.stdin.buffer,sys.stdout.buffer,lambda:None))')
    listener=socket.socket();listener.bind(('127.0.0.1',0));listener.listen()
    backend=RoboDojoBackend(cfg,worker_command=[sys.executable,'-u','-c',code])
    connections=[]
    def server():
        for _ in range(2):
            sock,_=listener.accept();c=socket_transport(sock);connections.append(c)
            serve_connection(backend,c)
    thread=threading.Thread(target=server,daemon=True);thread.start()
    host,port=listener.getsockname();remote=RemoteBackend(host=host,port=port)
    session=McpSession(remote,output_root=tmp_path/'logs')
    try:
        assert not session.call_tool('reset_task').get('isError')
        assert not session.call_tool('set_gripper',{'arm':'left','state':'close'}).get('isError')
        remote._connection.close()
        assert session.call_tool('observe')['isError'] and session.status=='aborted'
        assert not session.call_tool('reset_task').get('isError')
        assert session.step_index==0
        assert not session.call_tool('stop_episode').get('isError')
        summary=json.loads((session.journal.dir/'summary.json').read_text(encoding='utf-8'))
        assert summary['success_at_stop'] is True
    finally:
        session.close();listener.close();thread.join(5)
    assert not thread.is_alive() and backend.worker is None


def make_supervisor(tmp_path):
    from embodify_mcp.backend.robodojo import RoboDojoBackend
    config=tmp_path/'robodojo.json'
    config.write_text(json.dumps({'source':'unused','python':sys.executable,'output':str(tmp_path)}))
    code=('import sys;sys.path.insert(0,'+repr(str(Path(__file__).parent))+');'
          'from test_robodojo_backend import Driver;'
          'from embodify_mcp.robodojo.episode import EpisodeBackend;'
          'from embodify_mcp.backend.serve import serve_connection;'
          'from embodify_mcp.backend.transport import LineTransport;'
          'serve_connection(EpisodeBackend(Driver()),LineTransport(sys.stdin.buffer,sys.stdout.buffer,lambda:None))')
    return RoboDojoBackend(config,worker_command=[sys.executable,'-u','-c',code])


def test_archive_source_rejects_tampering(tmp_path):
    import hashlib
    from embodify_mcp.robodojo.source import verify_source
    from embodify_mcp.robodojo.config import REVISION,SUBMODULES
    f=tmp_path/'example.py';f.write_bytes(b'original')
    (tmp_path/'.embodify-source-manifest.json').write_text(json.dumps({
        'revisions':{'.':REVISION,**SUBMODULES},
        'sha256':{'example.py':hashlib.sha256(f.read_bytes()).hexdigest()}}))
    assert verify_source(tmp_path)==tmp_path.resolve()
    f.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='Changed source'): verify_source(tmp_path)


@pytest.mark.skipif(sys.platform!='linux',reason='Linux GPU worker lifecycle')
def test_parent_death_terminates_worker():
    import os,subprocess,time
    child=('from embodify_mcp.robodojo.process import bind_parent_lifetime;'
           'import os,time;bind_parent_lifetime();print(os.getpid(),flush=True);time.sleep(60)')
    parent=('import subprocess,sys,time;p=subprocess.Popen([sys.executable,"-u","-c",'+repr(child)+'],stdout=subprocess.PIPE,text=True);'
            'print(p.stdout.readline().strip(),flush=True);time.sleep(60)')
    proc=subprocess.Popen([sys.executable,'-u','-c',parent],stdout=subprocess.PIPE,text=True)
    try:
        pid=int(proc.stdout.readline());proc.terminate();proc.wait(timeout=5)
        for _ in range(100):
            stat=Path('/proc/%d/stat'%pid)
            if not stat.exists() or stat.read_text().split()[2]=='Z': break
            time.sleep(.02)
        else: raise AssertionError('GPU worker would survive supervisor death')
    finally:
        if proc.poll() is None: proc.kill();proc.wait(timeout=5)
