# Third-party notices

Original project code is distributed under Apache-2.0. Dependencies and
external assets retain their respective licenses.

## Included upstream source

`tests/fixtures/robodojo_run_eval.py` contains the result-writer method from
[RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo), commit
`726e9aabfaa642203722eb126f5eaf0f37f3e1ad`, `src/eval_client/eval_env.py`.
Copyright (c) 2025 Yue Chen. MIT license; the full notice is retained in that file.
It is a CPU test oracle, not a bundled simulator or a claim of GPU validation.

## External dependencies and assets

NumPy is a runtime dependency; pytest, build and twine are development tools.
Pillow and packaging are optional extras. These are installed separately and
not vendored. LIBERO, robosuite, MuJoCo, RoboDojo, Isaac Sim, Isaac Lab, cuRobo,
XPolicyLab and their transitive dependencies are also external installations.
Their own code, binary, dataset and asset terms apply independently.

No model weights, simulator asset archives, demonstrations, robot meshes,
textures, proprietary runtime wheels or original experiment recordings are
redistributed in this source tree. Installing an adapter does not grant rights
to redistribute those assets. Do not copy a whole simulator environment into a
release archive without reviewing its terms.
