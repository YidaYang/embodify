"""One episode's GPU owner. Heartbeats run independently of Isaac initialization."""
import argparse
import sys
from embodify_mcp.mcp import split_protocol_stdout
from embodify_mcp.backend.serve import serve_connection
from embodify_mcp.backend.transport import LineTransport
from embodify_mcp.robodojo.config import Settings
from embodify_mcp.robodojo.episode import EpisodeBackend

def main():
    from embodify_mcp.robodojo.process import bind_parent_lifetime
    bind_parent_lifetime()
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--task',help='Task this episode runs; default: the configured task')
    args=parser.parse_args()
    sink=split_protocol_stdout()
    from embodify_mcp.robodojo.isaac_driver import IsaacDriver
    backend=EpisodeBackend(IsaacDriver(Settings.load(args.config),args.task))
    connection=LineTransport(sys.stdin.buffer,sink.buffer,lambda:None)
    try: serve_connection(backend,connection,peer_timeout=30)
    finally: backend.close()

if __name__=='__main__':main()
