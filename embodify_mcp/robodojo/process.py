"""Linux worker lifetime is bounded by its supervisor's lifetime."""
import os
import signal
import sys


def bind_parent_lifetime():
    if sys.platform != 'linux': return
    import ctypes
    parent=os.getppid()
    libc=ctypes.CDLL(None,use_errno=True)
    # PR_SET_PDEATHSIG: reap GPU owner even if SSH/supervisor is force-terminated.
    if libc.prctl(1,signal.SIGTERM,0,0,0) != 0:
        raise OSError(ctypes.get_errno(),'prctl(PR_SET_PDEATHSIG)')
    if parent == 1 or os.getppid()!=parent:
        raise RuntimeError('Supervisor exited during worker startup')
