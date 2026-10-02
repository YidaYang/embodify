# Agent setup guide

This guide is for an AI agent that has been asked to install Embodify for
itself. Embodify gives you robot observation and control tools over MCP, plus
skills for using them. Work through the steps in order, keep the user informed,
and ask before installing system packages, using `sudo` or changing global
configuration.

If Embodify is already installed and the user asked you to connect a real
simulator, skip step 1 and start at step 2.

Names used throughout: the plugin is `embodify` from the `embodify` marketplace,
the Python package is `embodify-mcp`, the MCP server is registered as
`embodify`, and the server reads its settings from `~/.embodify/config.json`.

## 1. Install the plugin

The plugin brings the MCP server and the skills. Its server is started with
`uvx` from [uv](https://docs.astral.sh/uv/), which fetches the package from
PyPI. Check with `uvx --version`; if uv is missing, ask the user, then install
it with `python -m pip install uv` or the
[official installer](https://docs.astral.sh/uv/getting-started/installation/).

**Claude Code**

```sh
claude plugin marketplace add YidaYang/embodify
claude plugin install embodify@embodify
```

**Codex**

```sh
codex plugin marketplace add YidaYang/embodify
codex plugin add embodify@embodify
```

Remove older registrations so the tools do not appear twice: a server named
`embodify` added by hand (`claude mcp remove --scope user embodify` or
`codex mcp remove embodify`) and the former `embodify-skills` plugin.

**Other hosts:** register the server from [examples/mcp.json](../examples/mcp.json)
with your host's own mechanism, adding to the existing configuration rather than
replacing it. If the host supports Agent Skills (`SKILL.md` folders), copy the
folders in `plugin/skills/` of the repository into its skills directory.

Then fetch the package once and check it. This also makes the first server
start fast:

```sh
uvx --from embodify-mcp embodify-mcp-smoke
```

The smoke test needs no simulator or GPU. It prints a JSON summary containing
`"ok": true`.

## 2. Choose a backend

Unless the user already said, ask which they want:

- **Set up LIBERO** on this machine: continue with step 3.
- **Connect an existing simulator**, on this machine or a server: go to step 4.
- **Just try the tools** with the Fake backend: nothing to configure; go to step 6.

## 3. Set up LIBERO

LIBERO runs on Linux or WSL2. On Windows, run these commands inside WSL. The
MCP server keeps running next to you, and starts the LIBERO backend in its own
Python environment.

1. Install the system libraries (Ubuntu or Debian):

   ```sh
   sudo apt-get install -y libosmesa6 libegl1 libgl1 libglib2.0-0 build-essential
   ```

2. Clone the repository at the release that matches the server, where
   `<version>` is printed by `uvx --from embodify-mcp embodify-mcp --version`:

   ```sh
   git clone https://github.com/YidaYang/embodify.git
   cd embodify
   git checkout v<version>
   ```

3. Create a separate Python 3.8 environment, for example with
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

4. Write `~/.libero/config.yaml` **before** LIBERO is imported for the first
   time. Without it, LIBERO asks questions on standard input and the backend
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

5. Download the simulator assets (about 400 MB) into the `assets` directory:

   ```sh
   python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='jadechoghari/libero-assets', local_dir='/home/USER/.cache/libero/assets')"
   ```

   If Hugging Face is unreachable, set `HF_ENDPOINT` to a mirror such as
   `https://hf-mirror.com`. The download is complete when `wall.xml` exists in
   the assets directory.

6. Choose the renderer: `egl` on a machine with an NVIDIA GPU, otherwise
   `osmesa`. Set both `MUJOCO_GL` and `PYOPENGL_PLATFORM` to it, and run the
   LIBERO contract tests, which take about a minute:

   ```sh
   MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EMBODIFY_LIBERO_CONTRACT=1 \
     python -m pytest tests/test_backend_contract.py -k libero -q
   ```

   If `egl` fails, use `osmesa`.

7. Tell the server how to start the backend. Write `~/.embodify/libero.json`,
   following [examples/libero-local.json](../examples/libero-local.json), with
   the environment's `embodify-mcp-backend` and the renderer you chose. Backend
   options such as `--image-size` go at the end of `command`. With the agent on
   Windows and LIBERO in WSL, start the command through `wsl.exe`:

   ```json
   {
     "transport": "stdio",
     "command": ["wsl.exe", "-d", "Ubuntu-20.04", "--exec", "env",
                 "MUJOCO_GL=osmesa", "PYOPENGL_PLATFORM=osmesa",
                 "/home/USER/miniconda3/envs/embodify-libero/bin/embodify-mcp-backend",
                 "--backend", "libero"],
     "timeout_s": 300,
     "connect_timeout_s": 120,
     "heartbeat_timeout_s": 30
   }
   ```

   Then go to step 5 and set `"backend": "remote"` and
   `"remote-config": "libero.json"`.

## 4. Connect an existing simulator

Ask the user where the simulator runs and which backend it is.

- **Same machine:** install Embodify, at the server's version, into the
  simulator's environment. Write a transport file like
  [examples/libero-local.json](../examples/libero-local.json) whose `command`
  runs that environment's `embodify-mcp-backend` with the matching `--backend`
  option, then use it as the `remote-config` in step 5.
- **Another machine, such as a GPU server:** install Embodify and the simulator
  there, and make sure key-based SSH works without prompts. Copy
  [examples/remote-ssh.json](../examples/remote-ssh.json) to
  `~/.embodify/remote.json`, set the SSH alias, the remote Python and the
  backend options, and use it as the `remote-config` in step 5. Never put
  passwords in configuration files.
- **RoboDojo:** follow the RoboDojo section of [backend setup](backends.md),
  then set `"backend": "robodojo"` and `"robodojo-config"` in step 5. Worker
  startup can take minutes.

## 5. Write the settings file

The server reads `~/.embodify/config.json`, a JSON object whose keys are the
server's long options without the leading dashes; see
[backend setup](backends.md#server-options). Relative paths are resolved
against `~/.embodify/`. Update the existing file rather than replacing it. For
example:

```json
{
  "backend": "remote",
  "remote-config": "libero.json",
  "monitor-port": 8765
}
```

Run logs go to `~/.embodify/runs` unless `output-root` says otherwise.
`monitor-port` serves the live monitor at http://127.0.0.1:8765.

The plugin allows Codex 120 seconds to start the server and 900 seconds per
tool call. In Claude Code, if a slow simulator times out at startup, set the
`MCP_TIMEOUT` environment variable, in milliseconds, before starting Claude Code.

## 6. Finish

1. Tell the user to restart the session, or reload MCP servers, so the new tools appear.
2. After the restart, call `get_session_info` and `list_tasks`, then `reset_task`
   and `observe`, and check that the camera images show the scene.
3. Tell the user where run logs are written and, if you set `monitor-port`,
   where to open the monitor.
4. Start a robot profile with the `robot-profile` skill.
