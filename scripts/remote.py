"""Run commands on a remote GPU host and move files to it.

Built for the case where the only credential is a password: OpenSSH refuses to read a
password from a pipe, so paramiko is the way to script it.  The password is read from
an environment variable and is never written to disk, never echoed, and never included
in a committed file -- the host, port and user are the only things that appear in the
command line.

Usage::

    set REMOTE_PW=...                       # not in the repo, not in shell history you keep
    python scripts/remote.py probe
    python scripts/remote.py run -- "nvidia-smi -L"
    python scripts/remote.py put -- local/rel/path /root/remote/path
    python scripts/remote.py get -- /root/remote/path local/rel/path
    python scripts/remote.py sync -- data/processed/circuits

Batch mode reads a file of commands (one per line) so a whole probe or setup sequence
costs one connection instead of ten.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_HOST = "connect.cqa1.seetacloud.com"
DEFAULT_PORT = 38628
DEFAULT_USER = "root"

#: One round trip, so a probe does not pay connection setup ten times.
PROBE = r"""
set -x
hostname; uname -a
nvidia-smi -L || echo "NO_NVIDIA_SMI"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
nproc; free -g | head -2; df -h / /root 2>/dev/null | head -5
(which python3; python3 -V) 2>&1
python3 -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'n',torch.cuda.device_count())" 2>&1 | tail -2
python3 -c "import scipy,numpy,pandas,matplotlib;print('scipy',scipy.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'mpl',matplotlib.__version__)" 2>&1 | tail -2
python3 -c "import pyarrow;print('pyarrow',pyarrow.__version__)" 2>&1 | tail -1
ls -la /root 2>/dev/null | head -20
echo "PROBE_DONE"
"""


def _client(args):
    import paramiko

    pw = os.environ.get("REMOTE_PW")
    if not pw:
        raise SystemExit("set REMOTE_PW in the environment (password is never stored)")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(args.host, port=args.port, username=args.user, password=pw,
              timeout=30, banner_timeout=30, auth_timeout=30)
    return c


def _exec(client, cmd: str, *, timeout: int = 600, quiet: bool = False) -> tuple[int, str]:
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout, get_pty=False)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    text = out + (("\n[stderr]\n" + err) if err.strip() else "")
    if not quiet:
        print(text, end="" if text.endswith("\n") else "\n")
    return rc, text


def _no_proxy() -> None:
    """Make sure the transfer goes straight to the host.

    The shell here exports http_proxy/https_proxy pointing at a local port, and
    no_proxy covers only loopback.  paramiko does not consult them, but anything else
    in the process might, and a proxied bulk transfer is both slower and one more
    thing that can differ between the two machines.
    """
    for var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "ALL_PROXY"):
        os.environ.pop(var, None)
    os.environ["no_proxy"] = "*"
    os.environ["NO_PROXY"] = "*"


def _connect(args):
    import paramiko

    pw = os.environ.get("REMOTE_PW")
    if not pw:
        raise SystemExit("set REMOTE_PW in the environment (password is never stored)")
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(args.host, port=args.port, username=args.user, password=pw,
              timeout=30, banner_timeout=30, auth_timeout=30)
    t = c.get_transport()
    # Bigger window and packets: paramiko's default window makes a 13 MB file a long
    # series of round trips, which is what capped a single stream at ~0.5 MB/s on a
    # 31 ms link.
    t.default_window_size = 8 * 1024 * 1024
    t.default_max_packet_size = 128 * 1024
    import paramiko.sftp_file as _sf
    _sf.MAX_REQUEST_SIZE = 128 * 1024
    return c


def _sha256_local(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _connect_retry(args, *, attempts: int = 6, log=print):
    """Connect, backing off through sshd's start-up rate limit.

    Opening one connection per file (103 connections in a burst) tripped the server's
    MaxStartups limit and produced "Error reading SSH protocol banner" partway through.
    A persistent connection per worker avoids the burst entirely, and this retry covers
    the connections that are still needed.
    """
    import time

    import paramiko

    delay = 2.0
    last: Exception | None = None
    for i in range(attempts):
        try:
            return _connect(args)
        except (paramiko.SSHException, OSError, EOFError) as exc:
            last = exc
            if i == attempts - 1:
                break
            log(f"    connect retry {i + 1}/{attempts - 1} in {delay:.0f}s ({exc})")
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
    raise RuntimeError(f"could not connect after {attempts} attempts: {last}")


def _worker(args, queue, results: list, lock, progress: dict) -> None:
    """Hold one connection and drain the queue with it."""
    import posixpath
    import time

    c = _connect_retry(args)
    sftp = c.open_sftp()
    try:
        while True:
            with lock:
                if not queue:
                    return
                local, remote = queue.pop(0)
            t0 = time.time()
            size = local.stat().st_size
            # Count bytes as they move, not when a file finishes: with the biggest
            # files first, waiting for completions showed "0.0 MB" for minutes while
            # the link was actually running at full speed.
            state = {"sent": 0}

            def _cb(transferred, total, _s=state):
                delta = transferred - _s["sent"]
                _s["sent"] = transferred
                if delta:
                    with lock:
                        progress["bytes"] += delta

            for attempt in range(4):
                try:
                    _mkdirs(sftp, posixpath.dirname(remote) or ".")
                    sftp.put(str(local), remote, callback=_cb, confirm=True)
                    want = _sha256_local(local)
                    _, text = _exec(c, f"sha256sum {remote!r} | cut -d' ' -f1", quiet=True)
                    got = text.strip().splitlines()[-1].strip() if text.strip() else ""
                    if got != want:
                        raise RuntimeError(f"checksum mismatch {remote}: "
                                           f"{want[:16]} vs {got[:16] or '(none)'}")
                    break
                except Exception as exc:
                    if attempt == 3:
                        raise
                    # a dropped channel is recoverable: reconnect and retry this file
                    print(f"    retry {local.name} ({type(exc).__name__}: {exc})",
                          flush=True)
                    with lock:
                        progress["bytes"] -= state["sent"]
                        state["sent"] = 0
                    time.sleep(3 * (attempt + 1))
                    try:
                        c.close()
                    except Exception:
                        pass
                    c = _connect_retry(args)
                    sftp = c.open_sftp()
            with lock:
                results.append((local.name, size, time.time() - t0))
    finally:
        try:
            sftp.close()
            c.close()
        except Exception:
            pass


def cmd_upload(args) -> int:
    """Parallel, checksum-verified upload from a manifest of ``local|remote`` lines."""
    import threading
    import time

    _no_proxy()
    pairs: list[tuple[Path, str]] = []
    for line in Path(args.manifest).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        local, remote = line.split("|", 1)
        p = Path(local.strip())
        if not p.exists():
            raise SystemExit(f"missing local file: {p}")
        pairs.append((p, remote.strip()))

    total = sum(p.stat().st_size for p, _ in pairs)
    # Biggest first: with a fixed connection pool this finishes the long poles early
    # instead of leaving one 1 GB file to run alone at the end.
    pairs.sort(key=lambda pr: -pr[0].stat().st_size)
    print(f"uploading {len(pairs)} file(s), {total / 1e6:.1f} MB over "
          f"{args.workers} persistent connections, proxy bypassed")
    t0 = time.time()
    queue = list(pairs)
    results: list[tuple[str, int, float]] = []
    lock = threading.Lock()
    progress = {"bytes": 0}
    threads = [threading.Thread(target=_worker,
                                args=(args, queue, results, lock, progress),
                                daemon=True)
               for _ in range(args.workers)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            with lock:
                sent = progress["bytes"]
                n_done = len(results)
            rate = sent / 1e6 / max(time.time() - t0, 1e-9)
            eta = (total - sent) / 1e6 / max(rate, 1e-9)
            print(f"  {sent / 1e6:7.1f} / {total / 1e6:.1f} MB "
                  f"({100 * sent / max(total, 1):5.1f}%)  {n_done}/{len(pairs)} files  "
                  f"{rate:5.2f} MB/s  eta {eta / 60:4.1f} min", flush=True)
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("interrupted; already-uploaded files are checksum-verified and complete")
        return 1
    secs = time.time() - t0
    print(f"uploaded {len(results)}/{len(pairs)} files, "
          f"{sum(r[1] for r in results) / 1e6:.1f} MB in {secs:.1f}s "
          f"({sum(r[1] for r in results) / 1e6 / max(secs, 1e-9):.2f} MB/s aggregate), "
          f"all sha256-verified")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=os.environ.get("REMOTE_HOST", DEFAULT_HOST))
    ap.add_argument("--port", type=int, default=int(os.environ.get("REMOTE_PORT", DEFAULT_PORT)))
    ap.add_argument("--user", default=os.environ.get("REMOTE_USER", DEFAULT_USER))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    r = sub.add_parser("run")
    r.add_argument("command", nargs=argparse.REMAINDER)
    p = sub.add_parser("put")
    p.add_argument("local")
    p.add_argument("remote")
    g = sub.add_parser("get")
    g.add_argument("remote")
    g.add_argument("local")
    s = sub.add_parser("sync")
    s.add_argument("local_dir")
    s.add_argument("remote_dir")
    b = sub.add_parser("batch")
    b.add_argument("file")
    u = sub.add_parser("upload")
    u.add_argument("manifest")
    u.add_argument("--workers", type=int, default=8)
    sub.add_parser("close")
    args = ap.parse_args()

    if args.cmd == "close":
        return 0
    if args.cmd == "upload":
        return cmd_upload(args)

    client = _client(args)
    try:
        if args.cmd == "probe":
            _exec(client, PROBE, timeout=300)
            return 0

        if args.cmd == "run":
            cmd = " ".join(args.command).lstrip("- ").strip()
            if cmd.startswith("--"):
                cmd = cmd[2:].strip()
            rc, _ = _exec(client, cmd, timeout=7200)
            return rc

        if args.cmd == "batch":
            lines = [ln.strip() for ln in Path(args.file).read_text(encoding="utf-8").splitlines()]
            for ln in lines:
                if not ln or ln.startswith("#"):
                    continue
                print(f"\n$ {ln}")
                rc, _ = _exec(client, ln, timeout=7200)
                if rc != 0:
                    print(f"[exit {rc}]")
            return 0

        if args.cmd in ("put", "sync"):
            import posixpath

            sftp = client.open_sftp()
            local = Path(args.local if args.cmd == "put" else args.local_dir)
            remote = args.remote if args.cmd == "put" else args.remote_dir
            # Never ship bytecode caches: they are noise, they can be stale, and on
            # Windows their paths round-trip through Path() into backslashes.
            files = [local] if local.is_file() else [
                p for p in local.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
            ]
            total = sum(p.stat().st_size for p in files)
            print(f"uploading {len(files)} file(s), {total / 1e6:.1f} MB -> {remote}")
            done = 0
            for p in files:
                if local.is_file():
                    target = remote
                else:
                    target = remote.rstrip("/") + "/" + p.relative_to(local).as_posix()
                # posixpath, not Path: the remote side is POSIX and Path() would
                # rewrite the separators into Windows backslashes.
                _mkdirs(sftp, posixpath.dirname(target) or ".")
                try:
                    sftp.put(str(p), target)
                except Exception as exc:
                    print(f"  FAILED {p} -> {target}: {type(exc).__name__}: {exc}")
                    raise
                done += p.stat().st_size
                if len(files) > 20 and done % max(1, total // 10) < p.stat().st_size:
                    print(f"  {100 * done / max(total, 1):.0f}%")
            sftp.close()
            print("upload done")
            return 0

        if args.cmd == "get":
            sftp = client.open_sftp()
            local = Path(args.local)
            local.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(args.remote, str(local))
            sftp.close()
            print(f"downloaded {args.remote} -> {local}")
            return 0
    finally:
        client.close()
    return 0


def _mkdirs(sftp, path: str) -> None:
    """Create every missing parent, one component at a time.

    Naive ``Path.parts`` joining breaks on absolute POSIX paths: the leading "/" is a
    component, so cumulative joining produces ``//root/autodl-tmp`` and every mkdir
    then fails with ENOENT.
    """
    import posixpath

    parts = [p for p in path.split("/") if p]
    cur = "/" if path.startswith("/") else ""
    for part in parts:
        cur = posixpath.join(cur, part) if cur else part
        try:
            sftp.stat(cur)
        except IOError:
            try:
                sftp.mkdir(cur)
            except IOError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
