"""A fresh Isaac process per episode, outside the local MCP process."""
from pathlib import Path
from .base import Backend, BackendDisconnected
from .remote import RemoteBackend
from ..robodojo.config import Settings
from ..robodojo.catalogue import TaskCatalogue
from ..robodojo.episode import info
from ..robodojo.tasks import episode_budget

class RoboDojoBackend(Backend):
    def __init__(self, config, *, worker_command=None):
        self.config_path = str(Path(config).resolve())
        self.settings = Settings.load(self.config_path)
        self._catalogue = TaskCatalogue(self.settings.task, episode_budget_override=self.settings.episode_budget)
        # The configured task's budget; a chosen task carries its own in TaskSpec.episode_budget.
        self._info = info(episode_budget(self.settings.task, self.settings.episode_budget))
        self.worker_command = worker_command
        self.worker = None

    @classmethod
    def from_config(cls,path): return cls(path)
    @property
    def info(self): return self._info
    @property
    def catalogue(self): return self._catalogue

    def reset(self, task, variant):
        self.catalogue.commit(task,variant)
        self.end_episode()
        command = self.worker_command or [self.settings.python,'-m','embodify_mcp.robodojo.worker','--config',self.config_path]
        self.worker = RemoteBackend(command=list(command)+['--task',task.name],
                                    timeout_s=self.settings.startup_timeout_s, connect_timeout_s=30,
                                    heartbeat_timeout_s=30,close_timeout_s=10)
        try:
            if self.worker.info != info(task.episode_budget):
                raise RuntimeError('RoboDojo worker capabilities differ')
            self.worker.catalogue.commit(task,variant)
            snapshot = self.worker.reset(task,variant)
            self.catalogue.accept_instruction(self.worker.catalogue.task_payload()['instruction'])
            self.worker.timeout_s = self.settings.action_timeout_s
            return snapshot
        except Exception:
            self.end_episode()
            raise

    def _active(self):
        if self.worker is None: raise BackendDisconnected('remote_disconnected','The simulator process is unavailable; call reset_task')
        return self.worker
    def observe(self): return self._active().observe()
    def move_to(self,*args,**kwargs): return self._active().move_to(*args,**kwargs)
    def set_gripper(self,*args,**kwargs): return self._active().set_gripper(*args,**kwargs)
    def control_arms(self,*args,**kwargs): return self._active().control_arms(*args,**kwargs)
    def success(self): return self._active().success()
    def episode_audit(self): return self._active().episode_audit()
    def drain_transport_events(self): return [] if self.worker is None else self.worker.drain_transport_events()
    def journal_params(self):
        return {'robodojo':{'config':self.config_path,'action_unit':'take_action calls','integration':'native-eval-env-pilot',
                            'default_task':self.settings.task,'episode_budget':self.settings.episode_budget}}
    def end_episode(self):
        if self.worker is not None:
            worker,self.worker = self.worker,None
            worker.close()
