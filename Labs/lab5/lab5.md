# Lab 5 - Pipeline from git push to a running container

```mermaid
flowchart TD
    dev["Developer: git push / open PR<br/>(HUMAN)"] --> gh{{"GitHub Actions: ci.yml<br/>(AUTOMATIC on every push and PR)"}}
    gh --> test["job test<br/>uv sync, ruff, pytest"]
    test -->|pass| smoke["job smoke<br/>docker build, run with SKIP_MODEL, curl /health"]
    test -->|fail| red["run red: smoke + push skipped,<br/>PR cannot be merged"]
    smoke -->|pass, PR| review["Review + merge into main<br/>(HUMAN, allowed only when test and smoke are green)"]
    review --> mainpush
    smoke -->|pass, on main| mainpush["job push<br/>build + push ghcr.io/juleboost/mlops-lab-1:&lt;git SHA&gt;"]
    mainpush --> image[("ghcr image tagged with the commit SHA<br/>CODE VERSION DECIDED HERE")]

    data["data.py + validate_data.py<br/>then dvc add / dvc push (HUMAN, on the laptop)"] --> train["train.py on the laptop<br/>(HUMAN starts it, logs run to MLflow)"]
    train --> reg["register a new model version<br/>(HUMAN)"]
    reg --> gate["promote.py quality gate<br/>(HUMAN runs it, decision is AUTOMATIC)"]
    gate -->|"acc >= champion + 2*se"| champ[("alias champion moves<br/>MODEL VERSION DECIDED HERE")]
    gate -->|otherwise| chall["alias challenger, champion unchanged"]

    image --> deploy["docker compose pull / up or restart inference<br/>(HUMAN)"]
    champ --> deploy
    deploy --> running(["running container:<br/>image = git SHA, weights = models:/food11@champion<br/>resolved at startup"])
```

Automatic: everything inside GitHub Actions (lint, tests, image build, smoke test, push to ghcr on
main) and the promote/reject decision itself.

Still human: pushing and merging code, preparing data and running `dvc push`, training,
registering a version, running `promote.py`, and starting or restarting the container.

The code version is fixed by the git commit SHA, which is also the image tag. The model version
is fixed separately by the `champion` alias in the MLflow registry, read when the container starts.
