# Agent setup guide

This guide is for an AI agent that has been asked to install Embodify for
itself. Embodify gives you robot observation and control tools over MCP, plus
skills for using them. Work through the steps in order, keep the user informed,
and ask before installing system packages, using `sudo` or changing global
configuration.

If Embodify is already installed and the user asked you to connect a real
simulator, skip step 1 and start at step 2.

Names used throughout: the Python package and command are `embodify-mcp`, the
MCP server is registered as `embodify`, and the skills pack is `embodify-skills`.

## 1. Install Embodify

Install Embodify into a Python 3.10+ environment that your MCP host can launch:

```sh
git clone https://github.com/YidaYang/embodify.git
cd embodify
python -m pip install .
embodify-mcp-smoke
```

The smoke test needs no simulator or GPU. It prints a JSON summary containing
`"ok": true`.

## 2. Choose a backend

Unless the user already said, ask which they want:

- **Set up LIBERO** on this machine: continue with step 3.
- **Connect an existing simulator**, on this machine or a server: go to step 4.
- **Just try the tools** with the Fake backend (`--backend fake`): go to step 5.

## 3. Set up LIBERO

LIBERO runs on Linux or WSL2. On Windows, run these commands inside WSL and
launch the server through `wsl.exe` (see step 5).

1. Install the system libraries (Ubuntu or Debian):

   ```sh
   sudo apt-get install -y libosmesa6 libegl1 libgl1 libglib2.0-0 build-essential
   ```

2. Create a separate Python 3.8 environment, for example with
   `conda create -y -n embodify-libero python=3.8`. Run the following with that
   environment's Python, from the repository root:

   ```sh
   python -m pip install "pip<25"
   python -m pip install -r environments/libero-py38-constraints.txt
   python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
   python -m pip install --no-deps libero==0.1.1
   python -m pip install . pytest
   ```

   Install `libero` with `--no-deps`: its declared dependencies pull in large
   training packages that the simulator does not need.

3. Write `~/.libero/config.yaml` **before** LIBERO is imported for the first
   time. Without it, LIBERO asks questions on standard input and the MCP server
   hangs. Use absolute paths; `SITE` is the output of
   `python -c "import site; print(site.getsitepackages()[0])"`:

   ```yaml
   benchmark_root: SITE/libero/libero
   bddl_files: SITE/libero/libero/bddl_files
   init_states: SITE/libero/libero/init_files
   datasets: /home/USER/.libero/datasets
   assets: /home/USER/.cache/libero/assets
   ```

   Create the `datasets` directory; demonstration datasets are not needed.

4. Download the simulator assets (about 400 MB) into the `assets` directory:

   ```sh
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jadechoghari/libero-assets', local_dir='/home/USER/.cache/libero/assets')"
   ```

   If Hugging Face is unreachable, set `HF_ENDPOINT` to a mirror such as
   `https://hf-mirror.com`. The download is complete when `wall.xml` exists in
   the assets directory.

5. Choose the renderer: `egl` on a machine with an NVIDIA GPU, otherwise
   `osmesa`. Set both `MUJOCO_GL` and `PYOPENGL_PLATFORM` to it, and run the
   LIBERO contract tests, which take about a minute:

   ```sh
   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EMBODIFY_LIBERO_CONTRACT=1 \
     python -m pytest tests/test_backend_contract.py -k libero -q
   ```

   If `egl` fails, use `osmesa`.

## 4. Connect an existing simulator

Ask the user where the simulator runs and which backend it is.

- **Same machine:** install Embodify into the simulator's environment and point
  the MCP entry at that environment's `embodify-mcp`, with the matching
  `--backend` option.
- **Another machine, such as a GPU server:** install Embodify and the simulator
  there, and make sure key-based SSH works without prompts. Copy
  [examples/remote-ssh.json](../examples/remote-ssh.json), set the SSH alias,
  the remote Python and the backend options, then launch
  `embodify-mcp --backend remote --remote-config <file>` locally. Never put
  passwords in configuration files.
- **RoboDojo:** follow the RoboDojo section of [backend setup](backends.md).
  Worker startup can take minutes, so give the MCP server a long tool timeout.

## 5. Register the MCP server and skills

Register the server under the name `embodify` with your host's own mechanism,
adding to the existing configuration rather than replacing it. If `embodify` is
already registered, for example on the Fake backend, remove that entry first.
The launch command must not print anything to standard output, because that
stream carries the MCP protocol.

**Claude Code**

```sh
claude mcp remove --scope user embodify          # only if it is already registered
claude mcp add --scope user embodify -e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl   -- /path/to/env/bin/embodify-mcp --backend libero --output-root /path/to/runs
claude plugin marketplace add YidaYang/embodify
claude plugin install embodify-skills@embodify
```

**Codex**

```sh
codex mcp remove embodify                        # only if it is already registered
codex mcp add embodify --env MUJOCO_GL=egl --env PYOPENGL_PLATFORM=egl   -- /path/to/env/bin/embodify-mcp --backend libero --output-root /path/to/runs
codex plugin marketplace add YidaYang/embodify
codex plugin add embodify-skills@embodify
```

For slow simulators, add `startup_timeout_sec = 120` and `tool_timeout_sec = 900`
under `[mcp_servers.embodify]` in `~/.codex/config.toml`.

**Other hosts:** follow [examples/mcp.json](../examples/mcp.json) and your host's
documentation. If the host supports Agent Skills (`SKILL.md` folders), copy the
folders in `embodify-skills/skills/` into its skills directory.

**LIBERO in WSL with the agent on Windows:** register the `embodify` server with
`wsl.exe` as the command, and set the renderer inside it:

```json
{
  "command": "wsl.exe",
  "args": ["-d", "Ubuntu-20.04", "--", "bash", "-lc",
           "export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa && exec /home/USER/miniconda3/envs/embodify-libero/bin/embodify-mcp --backend libero --output-root /home/USER/embodify-runs"]
}
```

## 6. Finish

1. Tell the user to restart the session, or reload MCP servers, so the new tools appear.
2. After the restart, call `get_session_info` and `list_tasks`, then `reset_task`
   and `observe`, and check that the camera images show the scene.
3. Tell the user where run logs are written. Adding `--monitor-port 8765` to the
   server command serves the live monitor at http://127.0.0.1:8765.
4. Start a robot profile with the `robot-profile` skill.
