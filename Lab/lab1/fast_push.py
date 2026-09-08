r"""Fast replacement for `dvc push` against the DagsHub HTTP remote.

Does the same job as dvc push, but with much higher concurrency on the
existence check and live progress output, so it doesn't look frozen.

Reads the DagsHub token from the DAGSHUB_TOKEN environment variable.

Usage (PowerShell):
    $env:DAGSHUB_TOKEN = "<your token>"
    .\.venv\Scripts\python.exe fast_push.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

USER = "JuleBoost"
REMOTE = "https://dagshub.com/JuleBoost/mlops-lab-1.dvc/files/md5"
CACHE = Path(".dvc/cache/files/md5")
DIR_HASH = "36598702f6335bfb2b0b9510b7d1dfcb"

CHECK_CONCURRENCY = 128
UPLOAD_CONCURRENCY = 16


def hash_to_parts(md5: str) -> tuple[str, str]:
    return md5[:2], md5[2:]


def cache_path(md5: str) -> Path:
    prefix, rest = hash_to_parts(md5)
    return CACHE / prefix / rest


def remote_url(md5: str, suffix: str = "") -> str:
    prefix, rest = hash_to_parts(md5)
    return f"{REMOTE}/{prefix}/{rest}{suffix}"


async def check_one(session, sem, md5, state):
    async with sem:
        try:
            async with session.head(remote_url(md5), allow_redirects=True) as resp:
                exists = resp.status == 200
        except Exception:
            exists = False
    state["done"] += 1
    if not exists:
        state["missing"].append(md5)
    if state["done"] % 500 == 0:
        elapsed = time.time() - state["start"]
        rate = state["done"] / elapsed if elapsed else 0
        remaining = (state["total"] - state["done"]) / rate if rate else 0
        print(
            f"  checked {state['done']}/{state['total']}  "
            f"missing {len(state['missing'])}  "
            f"{rate:.0f}/s  eta {remaining/60:.1f} min",
            flush=True,
        )
    return exists


async def upload_one(session, sem, md5, state, suffix=""):
    path = cache_path(md5)
    if suffix:
        path = path.with_name(path.name + suffix)
    data = path.read_bytes()
    async with sem:
        for attempt in range(5):
            try:
                async with session.put(remote_url(md5, suffix), data=data) as resp:
                    if resp.status in (200, 201, 204):
                        state["uploaded"] += 1
                        print(
                            f"  uploaded {state['uploaded']}/{state['total_upload']}  {md5}{suffix}",
                            flush=True,
                        )
                        return True
                    last = f"HTTP {resp.status}"
            except Exception as exc:
                last = repr(exc)
            await asyncio.sleep(2 * (attempt + 1))
        state["failed"].append(md5 + suffix)
        print(f"  FAILED {md5}{suffix}: {last}", flush=True)
        return False


async def main():
    token = os.environ.get("DAGSHUB_TOKEN")
    if not token:
        sys.exit("Set DAGSHUB_TOKEN first:  $env:DAGSHUB_TOKEN = \"<token>\"")

    manifest = json.loads(cache_path(DIR_HASH).with_suffix(".dir").read_text())
    hashes = [entry["md5"] for entry in manifest]
    unique = sorted(set(hashes))
    print(f"manifest lists {len(hashes)} files ({len(unique)} unique hashes)\n")

    auth = aiohttp.BasicAuth(USER, token)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=60)
    connector = aiohttp.TCPConnector(limit=CHECK_CONCURRENCY, force_close=True)

    async with aiohttp.ClientSession(
        auth=auth, timeout=timeout, connector=connector
    ) as session:
        print("Phase 1: checking what's already on DagsHub...")
        state = {
            "done": 0,
            "total": len(unique),
            "missing": [],
            "start": time.time(),
        }
        sem = asyncio.Semaphore(CHECK_CONCURRENCY)
        await asyncio.gather(*(check_one(session, sem, h, state) for h in unique))

        missing = state["missing"]
        took = (time.time() - state["start"]) / 60
        print(f"\ncheck done in {took:.1f} min — {len(missing)} file(s) missing\n")

        if missing:
            print("Phase 2: uploading missing files...")
            up_state = {
                "uploaded": 0,
                "total_upload": len(missing),
                "failed": [],
            }
            up_sem = asyncio.Semaphore(UPLOAD_CONCURRENCY)
            await asyncio.gather(
                *(upload_one(session, up_sem, h, up_state) for h in missing)
            )
            if up_state["failed"]:
                print(f"\n{len(up_state['failed'])} file(s) still failed — rerun me.")
                return
            print("\nall missing files uploaded")
        else:
            print("nothing missing")

        print("\nPhase 3: uploading the .dir manifest...")
        man_state = {"uploaded": 0, "total_upload": 1, "failed": []}
        await upload_one(
            session, asyncio.Semaphore(1), DIR_HASH, man_state, suffix=".dir"
        )
        if man_state["failed"]:
            print("manifest upload failed — rerun me.")
            return

    print("\nDONE — data folder should now resolve on DagsHub.")


if __name__ == "__main__":
    asyncio.run(main())
