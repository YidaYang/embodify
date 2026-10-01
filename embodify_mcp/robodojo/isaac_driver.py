"""Isaac-only driver. Constructed lazily by an isolated P2 worker.

Uses RoboDojo's public evaluator implementation at the pinned revision. No policy
weights, task object poses or success signals are supplied to the MCP tool layer.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import uuid
from pathlib import Path
import numpy as np
from .config import REVISION
from .source import verify_source
from .tasks import COMMON_FILES, TASKS, episode_budget, native_budget, task_files

class LocalControlClient:
    """EvalEnv's policy reset hook; commands are supplied by our P2 worker."""
    def __init__(self, **kwargs): pass
    def call(self, func_name=None, **kwargs):
        if func_name != 'reset': raise RuntimeError('Unexpected policy-client call')
    def close(self): pass


class IsaacDriver:
    def __init__(self, settings, task=None):
        self.settings = settings
        self.task = settings.task if task is None else task
        if self.task not in TASKS: raise ValueError('Unknown RoboDojo task')
        self.app = self.env = None
        # Advertised in the worker handshake, i.e. before the environment exists.
        self.budget = episode_budget(self.task, settings.episode_budget)
        self.native_budget = native_budget(self.task)
        self._audit = None

    def reset(self):
        source = verify_source(self.settings.source, COMMON_FILES + task_files(self.task))
        if self.app is not None: raise RuntimeError('one episode per Isaac process')
        os.chdir(str(source))
        sys.path[:0] = [str(source), str(source/'XPolicyLab')]
        from isaaclab.app import AppLauncher
        parser = argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(['--headless','--enable_cameras','--device','cuda:0', '--kit_args', '--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera'])
        from env.camera_manager.capture.render_sync import add_zero_delay_kit_args
        add_zero_delay_kit_args(args)
        self.app = AppLauncher(args).app
        # Imports below require the running Kit application.
        from omegaconf import OmegaConf
        from utils.load_file import load_yaml
        from utils.pipeline_utils import process_config, process_randomization
        from src.eval_client import eval_env
        cfgroot = source/'env_cfg'
        evalcfg = load_yaml(str(cfgroot/'arx_x5.yml'))
        evalcfg.update(task_name=self.task,num_envs=1,device_id=0,eval_batch=False,
                       policy_name='mcp_agent_bridge',additional_info='action_type=ee',
                       seed=self.settings.seed,physx_monitor_enabled=False)
        cfg = OmegaConf.create({
            'sim':load_yaml(str(cfgroot/'sim/sim_config.yml')),
            'scene':load_yaml(str(cfgroot/'scene/default.yml')),
            'robot':load_yaml(str(cfgroot/'robot/dual_x5.yml')),
            'camera':load_yaml(str(cfgroot/'camera/camera_config.yml')),
            'task_env':load_yaml(str(source/'task/RoboDojo/config'/ (self.task+'.yml'))),
            'eval_cfg':evalcfg,
            'deploy_cfg':{'policy_name':'mcp_agent_bridge','host':'127.0.0.1','port':1},
        })
        cfg.sim.scene.num_envs = 1
        # Applies the task's robot, scene and physics settings from _task.yml.
        cfg, _ = process_config(process_randomization(cfg),task_name=self.task)
        cfg.sim.seed = [0]
        cfg.eval_cfg.eval_num = 1
        cfg.camera.default_frequency = cfg.eval_cfg.observation.collect_freq
        os.environ['ROBODOJO_RUN_ID'] = 'mcp-' + uuid.uuid4().hex
        # EvalEnv creates a WebSocket client unconditionally. Substitute only its
        # factory during construction, then restore; never edit upstream files.
        original = eval_env.WsModelClient
        try:
            eval_env.WsModelClient = LocalControlClient
            self.env = eval_env.create_eval_env(cfg,self.app)
        finally:
            eval_env.WsModelClient = original
        self.env.save_dir = str(Path(self.settings.output).resolve()/os.environ['ROBODOJO_RUN_ID'])
        self.env._stream_dir = str(Path(self.env.save_dir)/'_stream')
        layout = self.settings.layout_index
        if layout not in self.env.seed_manager.seed_info:
            raise RuntimeError('Configured layout does not exist')
        self.env.reset(seed=[layout])
        origin = self.env.env_origins[0].detach().cpu().numpy()
        if not np.allclose(origin, 0, atol=1e-8):
            raise RuntimeError("Pilot requires a single environment at world origin")
        self._apply_episode_budget()
        # Same order as run_eval before the policy loop, including the scripted support arm.
        self.env.run_reward()
        if hasattr(self.env,'get_score'): self.env.get_score()
        if getattr(self.env,'interact',False) and hasattr(self.env,'query_support_arm_traj'):
            self.env.query_support_arm_traj(env_idx=0)
        return self.observe()

    def _apply_episode_budget(self):
        # The task sets its native limit during reset. Override the live instance
        # afterwards, before the first action; leave the pinned upstream source intact.
        if int(self.env.step_lim) != self.native_budget:
            raise RuntimeError('Task step limit differs from the pinned catalogue')
        self.env.step_lim = self.budget

    def observe(self):
        raw = self.env.get_obs_batch(env_idx_list=[0],last_frame=True)[0]
        # Reduce only public camera images, preserving RGB channel order.
        from PIL import Image
        for cam in raw['vision'].values():
            rgb = cam.get('color')
            if rgb is not None:
                im = Image.fromarray(np.asarray(rgb)[:,:,:3])
                im.thumbnail((self.settings.image_size,self.settings.image_size))
                cam['color'] = np.asarray(im,dtype=np.uint8).copy()
        return raw

    @property
    def ended(self): return bool(self.env.end_flag[0])
    @property
    def success(self): return bool(self.env.end_flag[0] and self.env.success[0])

    def step(self, action):
        before = self.env.take_action_cnt[0]
        self.env.take_action(action)
        if self.env.take_action_cnt[0] != before+1:
            raise RuntimeError('Environment did not execute exactly one action')
        if self.env.unstable_envs: raise RuntimeError('Unstable simulation episode')
        return self.observe()

    def finish(self):
        if self._audit is not None: return self._audit
        env = self.env
        if env.unstable_envs: raise RuntimeError('Cannot report an unstable episode')
        # Explicit stop gets the same final-check predicates as the native limit.
        if not env.end_flag[0]:
            env.success[0] = bool(env.reward_manager.get_reward(final_check=True)[0] > 1-1e-3)
            env.end_flag[0] = True
        # run_eval normally captures running IDs BEFORE policy execution and then
        # writes results AFTER it. Our actions already ran: reproduce that captured
        # ID set, skip only policy execution / duplicate predicate registration.
        names = ['run_reward','eval_one_episode','get_running_env_idx_list']
        if hasattr(env,'get_score'): names.append('get_score')
        if hasattr(env,'query_support_arm_traj'): names.append('query_support_arm_traj')
        saved = {n:getattr(env,n) for n in names}
        try:
            for n in names: setattr(env,n,lambda **kwargs:None)
            env.get_running_env_idx_list = lambda:[0]
            env.run_eval()
        finally:
            for n, method in saved.items(): setattr(env,n,method)
        path = Path(env.save_dir)/'_result.json'
        result = json.loads(path.read_text())
        details = list(result['details'].values())
        if len(details)!=1 or bool(details[0]['success']) != self.success:
            raise RuntimeError('Official result differs from the episode outcome')
        self._audit = {'robodojo_result':result,'result_path':str(path),'upstream_revision':REVISION,
                       'episode_budget':self.budget,'native_episode_budget':self.native_budget,
                       'budget_overridden':self.budget != self.native_budget}
        return self._audit

    def close(self):
        try:
            if self.env is not None:
                self.env.model_client.close()
                self.env.close()
                self.env = None
        finally:
            if self.app is not None:
                self.app.close()
                self.app = None
