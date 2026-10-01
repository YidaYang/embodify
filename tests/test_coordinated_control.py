"""CPU contract and real-transport tests; these do not validate Isaac physics."""
import copy
import json
import time
from dataclasses import asdict
import numpy as np
import pytest
from embodify_mcp.backend.base import McpToolError
from embodify_mcp.backend.fake import FakeBackend
from embodify_mcp.backend.remote import RemoteBackend
from embodify_mcp.backend.stepped import quat_angle_between
from embodify_mcp.backend.wire import unpack_info
from embodify_mcp.mcp import McpSession
from embodify_mcp.robodojo.coordinated import slerp
from embodify_mcp.robodojo.episode import EpisodeBackend
from tests.test_robodojo_backend import Driver, make_supervisor, start
from tests.test_remote_backend import service
from tests.test_remote_latency import delayed_link


def goals(b, left=.1, right=.05):
    s = b.observe()
    return {n: {"pos": s.arms[n].pos + [d,0,0], "quat": s.arms[n].quat.copy()}
            for n,d in (("left",left),("right",right))}


class LaggingDriver(Driver):
    def __init__(self, gain=.25, drift=False):
        super().__init__()
        self.gain, self.drift, self.before = gain, drift, []
    def step(self, action):
        old = copy.deepcopy(self.raw)
        self.before.append(old)
        super().step(action)
        target = np.asarray(action["right_ee_pose"], dtype=float)
        current = np.asarray(old["state"]["right_ee_pose"], dtype=float)
        pose = target.copy()
        pose[:3] = current[:3] + self.gain*(target[:3]-current[:3])
        q = slerp(current[[4,5,6,3]],target[[4,5,6,3]],self.gain)
        pose[3:] = q[[3,0,1,2]]
        if self.drift and len(self.actions)==1: pose[1] += .03
        self.raw["state"]["right_ee_pose"] = pose.tolist()
        return self.observe()


def test_shared_translation_rotation_grippers_and_step_accounting():
    b=start(); target=goals(b)
    target["left"]["quat"]=np.array([0,0,np.sin(.3),np.cos(.3)])
    target["right"]["quat"]=np.array([np.sin(.15),0,0,np.cos(.15)])
    for v in target.values(): v["gripper"]="close"
    seen=[]
    r=b.control_arms(target,max_steps=30,on_step=lambda s:seen.append(s.step))
    assert r.stop_reason=="reached" and all(v["pose_reached"] for v in r.arm_results.values())
    assert seen==list(range(1,r.executed_steps+1))
    assert len(b.driver.actions)==r.executed_steps==r.snapshot.step
    assert all(v["gripper_command_applied"] for v in r.arm_results.values())
    for action,s in zip(b.driver.actions,r.telemetry["common_progress"]):
        assert action["left_ee_pose"][0]/.1==pytest.approx(s)
        assert (action["right_ee_pose"][0]-.5)/.05==pytest.approx(s)
        assert action["left_ee_joint_state"]==action["right_ee_joint_state"]==[0.]
    assert r.telemetry["common_progress"][-1]==1


def test_slow_arm_gates_progress_and_both_must_reach():
    d=LaggingDriver(); b=start(d); target=goals(b,.05,.05)
    target["right"]["quat"]=np.array([0,0,np.sin(.15),np.cos(.15)])
    r=b.control_arms(target,max_steps=100)
    assert r.stop_reason=="reached"
    progress=r.telemetry["common_progress"]
    assert any(a==c for a,c in zip(progress,progress[1:]))
    assert all(v["pose_reached"] for v in r.arm_results.values())
    for action,old in zip(d.actions,d.before):
        for n in ("left","right"):
            pose=np.asarray(action[n+"_ee_pose"]); before=np.asarray(old["state"][n+"_ee_pose"])
            assert np.linalg.norm(pose[:3]-before[:3])<=.020001
            assert quat_angle_between(pose[[4,5,6,3]],before[[4,5,6,3]])<=.100001


def test_jammed_arm_stops_both_without_other_arm_running_ahead():
    b=start(LaggingDriver(gain=0)); r=b.control_arms(goals(b),max_steps=30)
    assert r.stop_reason=="stalled" and r.executed_steps==6
    assert len(set(r.telemetry["common_progress"]))==1
    assert r.snapshot.arms["left"].pos[0]<=.016001
    assert not r.arm_results["right"]["pose_reached"]


def test_unspecified_arm_holds_initial_pose_after_disturbance():
    b=start(LaggingDriver(gain=1,drift=True)); initial=b.observe().arms["right"]
    target=goals(b,.08); target.pop("right")
    r=b.control_arms(target,max_steps=30)
    assert r.stop_reason=="reached"
    assert r.telemetry["common_progress"][0]==r.telemetry["common_progress"][1]
    # Disturbance recovery targets may be intermediate, but the hold anchor never drifts.
    assert b.driver.actions[-1]["right_ee_pose"] == list(initial.pos)+[1,0,0,0]
    for action, old in zip(b.driver.actions, b.driver.before):
        before = np.asarray(old["state"]["right_ee_pose"][:3])
        assert np.linalg.norm(np.asarray(action["right_ee_pose"][:3])-before) <= .020001
    assert np.allclose(r.snapshot.arms["right"].pos,initial.pos)


def test_gripper_only_one_step_then_noop():
    b=start(); target=goals(b,0,0)
    for v in target.values(): v["gripper"]="close"
    r=b.control_arms(target,max_steps=30)
    assert r.stop_reason=="command_applied" and r.executed_steps==1
    r=b.control_arms(target,max_steps=30)
    assert r.stop_reason=="no_op" and r.executed_steps==0
    assert not any(v["gripper_command_applied"] for v in r.arm_results.values())
    assert all(a.gripper_opening is None for a in r.snapshot.arms.values())


def test_zero_budget_and_late_invalid_arm_never_change_latches():
    b=start(); target=goals(b); target["left"]["gripper"]="close"
    r=b.control_arms(target,max_steps=0)
    assert r.stop_reason=="limit" and r.executed_steps==0
    target["right"]["quat"]=[0,0,0,0]
    with pytest.raises(McpToolError): b.control_arms(target,max_steps=30)
    assert not b.driver.actions and b._commands=={"left":"open","right":"open"}


@pytest.mark.parametrize("args",[
    {},{"arms":{}},{"arms":{"left":{}}},{"arms":{"both":{"gripper":"close"}}},
    {"arms":{"left":{"delta_xyz":[True,0,0]}}},
    {"arms":{"left":{"delta_xyz":[float("nan"),0,0]}}},
    {"arms":{"left":{"gripper":"close"},"right":{"gripper":"settled"}}},
    {"arms":{"left":{"delta_xyz":[0,0]}}},
    {"arms":{"left":{"delta_xyz":[0,0,0],"seed":1}}},
    {"arms":{"left":{"gripper":"close"}},"max_steps":500},
])
def test_mcp_validates_entire_request_before_execution(tmp_path,args):
    b=EpisodeBackend(Driver()); s=McpSession(b,output_root=tmp_path)
    try:
        assert not s.call_tool("reset_task").get("isError")
        r=s.call_tool("control_arms",args)
        assert r["isError"] and r["structuredContent"]["error"]=="invalid_input"
        assert not b.driver.actions and b._commands=={"left":"open","right":"open"}
    finally: s.close()


def test_capability_schema_budget_log_and_public_feedback(tmp_path):
    fake=McpSession(FakeBackend(image_size=16),output_root=tmp_path/"fake")
    assert "control_arms" not in {t["name"] for t in fake.tools()}
    fake.close()
    b=EpisodeBackend(Driver()); s=McpSession(b,output_root=tmp_path,max_steps_per_call=2)
    try:
        tool=next(t for t in s.tools() if t["name"]=="control_arms")
        assert set(tool["inputSchema"]["properties"]["arms"]["properties"])=={"left","right"}
        assert not s.call_tool("reset_task").get("isError")
        args={"arms":{"left":{"delta_xyz":[.1,0,0]},"right":{"delta_xyz":[.05,0,0]}}}
        r=s.call_tool("control_arms",args); assert not r.get("isError"),r
        data=r["structuredContent"]
        assert data["action"]["stop_reason"]=="step_cap"
        assert data["action"]["executed_steps"]==data["step"]==2 and data["steps_left"]==548
        assert len([c for c in r["content"] if c["type"]=="image"])==3
        assert all(word not in json.dumps(r) for word in b.info.forbidden_terms)
        assert data["action"]["arms"]["left"]["remaining_delta_xyz"][0]==pytest.approx(.068)
        assert any(e["kind"]=="mcp_control_arms" for e in s.journal.read_events())
        b._step=549; r=s.call_tool("control_arms",args)
        assert r["structuredContent"]["action"]["stop_reason"]=="budget"
        assert r["structuredContent"]["step"]==550
        r=s.call_tool("control_arms",{"arms":{"left":{"gripper":"close"}}})
        assert r["isError"] and b._commands["left"]=="open"
    finally: s.close()


def test_shortest_rotation_and_ended_environment():
    b=start(); target=goals(b,0,0); target["left"]["quat"]=[0,0,0,-1]
    assert b.control_arms(target,max_steps=30).stop_reason=="no_op"
    b.driver.ended=True
    r=b.control_arms(goals(b),max_steps=30)
    assert r.stop_reason=="ended" and r.executed_steps==0


def test_nested_remote_worker_joint_action_and_reset(tmp_path):
    backend=make_supervisor(tmp_path)
    with service(backend,frame_stride=3) as (remote,_):
        s=McpSession(remote,output_root=tmp_path/"logs")
        try:
            for _ in range(2):
                assert not s.call_tool("reset_task").get("isError")
                r=s.call_tool("control_arms",{"arms":{
                    "left":{"delta_xyz":[.04,0,0],"gripper":"close"},
                    "right":{"delta_xyz":[0,0,.04],"gripper":"close"}}})
                assert not r.get("isError"),r
                assert r["structuredContent"]["action"]["stop_reason"]=="reached"
                assert r["structuredContent"]["state"]["arms"]["right"]["gripper"]["command"]=="close"
                assert any(e["kind"]=="mcp_control_arms" for e in s.journal.read_events())
                assert not s.call_tool("stop_episode").get("isError")
        finally: s.close()
    assert backend.worker is None


def test_corrupt_feedback_aborts_instead_of_exposing_fields(tmp_path):
    class Corrupt(EpisodeBackend):
        def control_arms(self,*args,**kwargs):
            r=super().control_arms(*args,**kwargs); r.arm_results["left"]["reward"]=123; return r
    with service(Corrupt(Driver())) as (remote,_):
        s=McpSession(remote,output_root=tmp_path)
        try:
            s.call_tool("reset_task")
            r=s.call_tool("control_arms",{"arms":{"left":{"delta_xyz":[.04,0,0]}}})
            assert r["isError"] and s.status=="aborted" and "reward" not in json.dumps(r)
        finally: s.close()


def test_older_capability_defaults_to_disabled():
    data=asdict(FakeBackend(image_size=16).info); data.pop("supports_coordinated_control")
    assert not unpack_info(data).supports_coordinated_control


def test_two_second_rtt_joint_control_is_one_motion_rpc(tmp_path):
    class Timed(EpisodeBackend):
        count=0; duration=None
        def control_arms(self,*args,**kwargs):
            self.count+=1; start=time.monotonic()
            r=super().control_arms(*args,**kwargs); self.duration=time.monotonic()-start; return r
    backend=Timed(Driver())
    with service(backend) as (initial,_):
        initial.close()
        with delayed_link(initial.host,initial.port) as (host,port):
            remote=RemoteBackend(host=host,port=port,frame_stride=5); s=McpSession(remote,output_root=tmp_path)
            try:
                assert not s.call_tool("reset_task").get("isError")
                before=time.monotonic()
                r=s.call_tool("control_arms",{"arms":{
                    "left":{"delta_xyz":[1,0,0]},"right":{"delta_xyz":[.5,0,0]}}})
                elapsed=time.monotonic()-before
                assert not r.get("isError"),r
                assert r["structuredContent"]["action"]["executed_steps"]==30
                assert backend.count==1 and backend.duration<3 and 4<=elapsed<15
                assert not s.call_tool("stop_episode").get("isError")
            finally: s.close()


def test_joint_timeout_aborts_worker_without_replaying_action(tmp_path):
    import threading
    class SlowDriver(Driver):
        def __init__(self):
            super().__init__()
            self.released = threading.Event()
        def step(self, action):
            time.sleep(.05)
            return super().step(action)
        def close(self):
            super().close()
            self.released.set()
    d = SlowDriver()
    with service(EpisodeBackend(d), timeout_s=.12) as (remote, _):
        s = McpSession(remote, output_root=tmp_path)
        try:
            assert not s.call_tool("reset_task").get("isError")
            r = s.call_tool("control_arms", {"arms": {
                "left": {"delta_xyz": [1,0,0]}, "right": {"delta_xyz": [.5,0,0]}}})
            assert r["isError"] and r["structuredContent"]["error"] == "remote_timeout"
            assert s.status == "aborted" and d.released.wait(3)
            assert 1 <= len(d.actions) < 30
            calls = [e for e in s.journal.read_events() if e["kind"] == "tool_call"]
            assert calls
        finally: s.close()


def test_remote_invalid_second_target_is_atomic_and_episode_remains_usable():
    d = Driver()
    with service(EpisodeBackend(d)) as (remote, _):
        task, variant = remote.catalogue.resolve({}, locked=False)
        remote.reset(task, variant)
        target = goals(remote)
        target["left"]["gripper"] = "close"
        target["right"]["quat"] = [0,0,0,0]
        with pytest.raises(McpToolError, match="Joint target"):
            remote.control_arms(target, max_steps=30)
        assert not d.actions and remote.observe().step == 0
        assert remote.control_arms(goals(remote), max_steps=30).stop_reason == "reached"


def test_rotation_only_mcp_and_environment_ending_during_joint_action(tmp_path):
    class Ending(Driver):
        def step(self, action):
            result = super().step(action)
            self.ended = True
            return result
    s = McpSession(EpisodeBackend(Ending()), output_root=tmp_path)
    try:
        s.call_tool("reset_task")
        r = s.call_tool("control_arms", {"arms": {
            "left": {"delta_rpy": [0,0,.4]}, "right": {"delta_rpy": [0,.2,0]}}})
        assert not r.get("isError"), r
        action = r["structuredContent"]["action"]
        assert action["stop_reason"] == "ended" and action["executed_steps"] == 1
        assert action["arms"]["left"]["remaining_angle_rad"] > .05
        assert action["arms"]["left"]["achieved_delta_xyz"] == [0,0,0]
    finally: s.close()
