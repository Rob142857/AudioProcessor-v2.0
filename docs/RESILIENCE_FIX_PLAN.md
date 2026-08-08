# Pipeline resilience — failure analysis and fix plan

Investigated: 2026-08-08. Target implementer: Sonnet 5.

Run analysed: `C:\Users\RobertEvans\OneDrive - RME Solutions Technology\_PG Completed Recordings 84-97 - Polished`
(30 GLM review workers + 1 Parakeet GPU worker, cancelled by the user at the end).

---

## 1. What actually happened

Evidence: 1730 per-job `manifest.json` files under the polished root, scanned recursively.
Job directories sit at two different depths — `<year>/<job>__mp3` for most, and
`Tapes From Joe (MW, RL, et al)/<year>/<job>__mp3` for 98 of them. Any audit script must
use `rglob`/`glob(recursive=True)`, not a fixed `*/*/` depth.

| status | count |
|---|---|
| `needs_review` | 985 |
| `verified` | 522 |
| **`failed`** | **221** |
| `running` (stuck) | 1 |
| `cancelled` | 1 |

Parakeet was fine. **218 of the 221 failures still have a valid `raw.txt`** — the GPU
transcription completed and is durable. Everything failed *after* the raw boundary,
in the GLM cleanup stage or in manifest persistence.

### Failure breakdown

| # | Cause | Class |
|---|---|---|
| **201** | Windows `MAX_PATH` exceeded creating `cleanup-chunks/<sha256>/` | **A** |
| 9 | DNS `getaddrinfo failed` — cleanup retries exhausted | **B** |
| 9 | `PermissionError [WinError 5]` renaming `manifest.json` | **C** |
| 2 | `ParakeetError` (1 worker timeout, 1 too-short audio) | **D** |
| 1 | job stuck in `running`, never reaped | **E** |

Class A is **91% of all failures**. Every `— failed.` line in the operator log traces to
it; every `— needs_review.` line succeeded. Spot-checked against the log directly:

| log line | manifest |
|---|---|
| `1985 0122 Tibetan Book of the Dead 1.mp3 — failed` | `failed`, WinError 206 |
| `1985 0129 Tibetan Book of the Dead 2.mp3 — failed` | `failed`, WinError 206 |
| `1985 0326 Egyptian Studies 01.mp3 — failed` | `failed`, WinError 206 |
| `1985 0730 Egyptian Studies.mp3 — needs_review` | `needs_review`, no error |
| `1985 0917 Gardens.mp3 — needs_review` | `needs_review`, no error |

---

## 2. Root cause A — MAX_PATH (201 of 221 failures, 91%)

Two error strings, one bug:

```
FileNotFoundError: [WinError 206] The filename or extension is too long: '...\cleanup-chunks\b367829...'   (140x)
FileNotFoundError: [WinError 3] The system cannot find the path specified: '...\cleanup-chunks\b367829...'   (61x)
```

Both point at a directory path with no filename — that is `mkdir`, not a file write.

**The exact line:** [cleanup_client.py:614](cleanup_client.py:614)

```python
run_dir = Path(checkpoint_dir) / input_sha256
run_dir.mkdir(parents=True, exist_ok=True)     # <-- plain path, no \\?\ prefix
```

The codebase **already has the fix helper** — `_windows_extended_path()` at
[cleanup_client.py:62](cleanup_client.py:62) — and already uses it for checkpoint
*file* reads and writes at [cleanup_client.py:913](cleanup_client.py:913) and
[cleanup_client.py:1009](cleanup_client.py:1009). It was simply never applied to the
`mkdir` that creates the directory those files live in. The directory creation fails
first, so the protected file paths never get a chance to run.

**Why the paths are long.** The output root is 89 chars before any job name:

```
C:\Users\RobertEvans\OneDrive - RME Solutions Technology\_PG Completed Recordings 84-97 - Polished\
  1993 - Group\
  L0193 The Inner Wisdom Dynamics of Arcane Christianity 20.1.1993_mixdown_Mono__flac\
  cleanup-chunks\
  b367829472cded67c4ae43762829ce69c5c8eed13bfc1d18f37b34b104b5d04f      <-- 64 chars
= 275 characters
```

`MAX_PATH` is 260. `HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled`
is **`0`** on this machine, so Python's plain-path calls are capped at 260.

**This will keep getting worse.** Across all 1730 jobs, the projected `run_dir` length
distribution is:

| length | jobs |
|---|---|
| 180–239 | 1429 |
| 240–259 | 239 (within 12 chars of the limit) |
| **260+** | **62 (already fatal)** |

202 jobs sit close enough that adding one word to a lecture title, or moving the root
one folder deeper, tips them over. The longest is 286 characters
(`1995 Prepared\Lecture 26 120795 Some Correspondences nof Respiration Hydrogen and Carbon_mixdown_Mono__flac`).

The `Tapes From Joe (MW, RL, et al)\<year>\` subtree is a whole extra nesting level, which
is why its failure rate is worse than the rest: 53 of its 98 jobs failed.

---

## 3. Root cause B — network retry ceiling too low (9 failures)

```
CleanupNetworkError: cleanup request failed after 4 attempts: <urlopen error [Errno 11001] getaddrinfo failed>
```

[cleanup_client.py:449](cleanup_client.py:449): `max_attempts=4`, `retry_base_delay=1.0`,
exponential. Total tolerance = 1 + 2 + 4 = **~7 seconds**.

`getaddrinfo failed` is DNS, not the server. A laptop DNS blip, a VPN re-key, or a Wi-Fi
roam routinely lasts longer than 7 seconds. With 30 workers hammering concurrently, any
such blip takes out every job that was mid-request. Nine jobs died to a fault that would
have cleared on its own.

DNS/connection failures should also be treated differently from HTTP 5xx: a 5xx means the
server is unhappy and backing off hard is right; `getaddrinfo` means the *client* has no
network and should wait for it to come back.

---

## 4. Root cause C — non-atomic-enough atomic write (9 failures)

```
PermissionError: [WinError 5] Access is denied:
  '...\0827 Behind the thinking process__mp3\.manifest.json.a7d3wjdj.tmp'
  -> '...\0827 Behind the thinking process__mp3\manifest.json'
```

[archive_pipeline.py:186](archive_pipeline.py:186):

```python
os.replace(temporary, path)     # single attempt, no retry
```

On POSIX `rename()` over an open file is fine. On Windows it fails with `WinError 5`
if *anything* has the destination open — and the output root is inside **OneDrive**,
which opens files to sync them, plus Defender scans on close. This is a transient
lock measured in milliseconds. One retry would have caught all nine.

Note this also loses work in a nastier way than A or B: the job had *completed* and was
being written down when the rename failed, so a full GLM review was thrown away.

---

## 5. Root causes D and E (3 failures)

- `ParakeetError: Parakeet worker timed out; it was stopped safely` — no retry path;
  a single timeout is terminal for that recording.
- `ParakeetError: ValueError: normalize_batch with 'per_feature' ... received a tensor of
  length 1` — the audio is too short for one mel frame. This is a bad/near-empty input,
  not a transient fault. It needs a clear pre-flight rejection, not a NeMo stack trace.
- One job left in `status: "running"`, `stage: "cleaning"`
  (`1994 Prepared\Lecture 4 The Creative Influence of Light 9.2.1994_mixdown_Mono__flac`).
  The process died between `manifest["status"] = "running"` and the outcome write. Nothing
  reaps it, so it is neither retried nor reported as failed — it is silently invisible.

---

## 6. Cross-cutting gap — no in-run retry at all

[archive_pipeline.py:2399-2415](archive_pipeline.py:2399): `process_one` catches every
exception, writes `status: "failed"`, and returns. The GLM worker records the result and
moves to the next item. **Nothing transient is ever re-attempted inside a run.**

A 7-second DNS blip and a permanently malformed audio file are treated identically. All
201 MAX_PATH failures were deterministic and correctly non-retryable — but the 20 in
classes B/C/D were all transient, and nearly all would have passed on a second attempt.

Good news, and the reason this is recoverable: `FINAL_STATUSES` is
`{"verified", "needs_review"}` ([archive_pipeline.py:77](archive_pipeline.py:77)), so
`failed` jobs *are* re-attempted on the next run, and completed cleanup chunks are reused
from the checkpoint dir. Once A–C are fixed, a plain re-run recovers all 221 without
re-transcribing anything.

---

## 7. Fix plan

Ordered by impact. Each item is independently shippable.

### Fix 1 — Extended paths for checkpoint directory creation *(fixes 201 / 91%)*

`cleanup_client.py`, in `cleanup_text` around line 613:

```python
run_dir: Path | None = None
if checkpoint_dir is not None:
    run_dir = Path(checkpoint_dir) / input_sha256
    Path(_windows_extended_path(run_dir)).mkdir(parents=True, exist_ok=True)
```

Keep `run_dir` itself as the plain path — the downstream `_load_checkpoint` /
`_save_checkpoint` already apply `_windows_extended_path` themselves, so wrapping it
twice is unnecessary and the helper is already idempotent (it short-circuits on a
`\\?\` prefix).

Audit for any other bare `mkdir` / `exists` / `open` on a checkpoint-derived path in
`cleanup_client.py` and apply the same treatment.

**Test:** extend `tests/test_cleanup_client.py` (see the existing `\\?\` assertion at
line 306). Add a case with a `checkpoint_dir` long enough that
`len(checkpoint_dir / sha256) > 260`, assert `cleanup_text` completes and the checkpoint
files land. Skip on non-Windows.

### Fix 2 — Retrying atomic replace *(fixes 9)*

`archive_pipeline.py`, `atomic_write_bytes` at line 175. Wrap the `os.replace` in a short
bounded retry, and use extended paths on Windows for defence in depth:

```python
for attempt in range(5):
    try:
        os.replace(temporary, path)
        break
    except PermissionError:
        if attempt == 4:
            raise
        time.sleep(0.1 * (2 ** attempt))   # 0.1, 0.2, 0.4, 0.8s — ~1.5s total
```

Only retry `PermissionError`/`OSError` with `winerror in (5, 32)`. Do not retry anything
else — a genuine permissions problem should still fail loudly rather than stall.

Add the same `_windows_extended_path` treatment here (lift the helper into a shared module,
or duplicate it — it is 8 lines and `archive_pipeline.py` should not import from
`cleanup_client.py` just for this).

**Test:** monkeypatch `os.replace` to raise `PermissionError(winerror=5)` twice then
succeed; assert the file is written and `os.replace` was called 3 times.

### Fix 3 — Network retry budget and DNS-aware backoff *(fixes 9)*

`cleanup_client.py`:

- Raise the default `max_attempts` from 4 to **6** and `retry_base_delay` from 1.0 to
  **2.0** → tolerance goes from ~7s to ~2 minutes.
- Cap individual backoff at ~30s so a `Retry-After` header can't stall a worker for hours.
- Classify DNS/connection-refused (`socket.gaierror`, `ConnectionRefusedError`,
  `[Errno 11001]`) separately from HTTP 5xx: for those, use a **flat** ~10s poll rather
  than exponential — you are waiting for a link to come back, not easing load on a server.

Make both values configurable via `PipelineConfig` so a batch of 1600 over a flaky link
can be tuned without a code change.

**Test:** existing tests inject `transport` and `sleep`; add a case where the transport
raises `URLError(gaierror)` five times then succeeds, and assert the call succeeds with
the expected sleep sequence.

### Fix 4 — Retry transient failures within a run *(the structural fix)*

In `process_one`'s `except Exception` handler ([archive_pipeline.py:2399](archive_pipeline.py:2399)),
classify before writing `failed`:

- **Transient** — `CleanupNetworkError`, `PermissionError`, `OSError` with winerror 5/32,
  Parakeet timeout, HTTP 429/5xx. Record `status: "retry_pending"`, increment a
  `transient_attempts` counter, and re-queue the source once (cap at 2 in-run retries).
- **Permanent** — everything else, including MAX_PATH. Write `failed` as today.

The GLM worker loop ([archive_pipeline.py:2577](archive_pipeline.py:2577)) puts the source
back on `cleanup_queue` when it sees `retry_pending`. Guard against an infinite loop with
the per-job counter, not a global one. Make sure `retry_pending` is **not** added to
`FINAL_STATUSES` so a crash mid-retry still resolves on the next run.

### Fix 5 — Reap stale `running` jobs *(fixes 1, prevents silent loss)*

At run start, scan for manifests with `status: "running"` whose `mtime` is older than a
threshold (say 2 hours) and rewrite them to `failed` with
`error: "abandoned; process exited during <stage>"`. Without this they are invisible to
both the retry path and the summary counts.

### Fix 6 — Parakeet input pre-flight *(fixes 1, improves diagnostics)*

Before dispatching to the model, reject audio shorter than the mel hop length with a clear
message (`"audio too short to transcribe: 0.03s"`) rather than surfacing a NeMo
`normalize_batch` `ValueError`. Check `parakeet_stt.py` / `parakeet_worker.py` — a duration
probe already exists for the coverage check in `stt_coverage.py` and can be reused.

Separately, treat "Parakeet worker timed out" as transient under Fix 4.

### Fix 7 — Operator-facing failure summary

At the end of a run, print a grouped tally instead of leaving the operator to read 1600
manifests:

```
221 failed:
  201  path too long (Windows MAX_PATH)  — see docs/RESILIENCE_FIX_PLAN.md
    9  cleanup network unreachable
    9  manifest write blocked (OneDrive/AV lock)
    2  Parakeet error
```

This run's failures were only diagnosable by scripting over the manifests. The GUI showed
"failed" 221 times with no reason attached.

---

## 8. Recommended operational changes (no code)

Independent of the fixes, both are worth doing:

1. **Enable long paths on this machine.** Fix 1 makes the pipeline correct regardless, but
   this removes a whole class of Windows tooling pain:
   ```bash
   reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f
   ```
   Requires an elevated prompt and a reboot.

2. **Do not use a synced OneDrive folder as the working output root.** OneDrive's sync
   handles caused all 8 `WinError 5` failures, it uploads every intermediate chunk
   checkpoint, and its long root path is 89 of the 260 characters. Write to a local
   working root and publish the finished `.docx` files into OneDrive as the last step.

---

## 9. Recovery of this run

After Fixes 1–3 land, re-run the same command over the same root. `failed` is not a final
status, so all 221 are re-attempted; `raw.txt` is intact for 218 of them so Parakeet does
not re-run; completed cleanup chunks are reused from `cleanup-chunks/`. Expected cost is
GLM review only, on ~221 recordings.

Verify afterwards with the command below, run from the polished root. Note
`recursive=True` — the `Tapes From Joe` subtree is one level deeper, and a `*/*/` glob
silently under-reports by 98 jobs and 53 failures:

```bash
python -c "import json,glob,collections; print(collections.Counter(json.load(open(p,encoding='utf-8')).get('status') for p in glob.glob('**/manifest.json',recursive=True)))"
```
