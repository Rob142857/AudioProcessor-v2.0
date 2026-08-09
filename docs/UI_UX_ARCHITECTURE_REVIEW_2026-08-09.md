# UI/UX and architecture review — operator experience

Reviewed: 2026-08-09. Scope: `gui_transcribe.py`, `gui_components.py`, `progress_window.py`,
`run.bat`, `pipeline_control.py`, and the operator-facing progress strings in
`archive_pipeline.py`. Report only — no `.py` files were modified.

Context: this pipeline was heavily hardened this week (run lock, stale-`running` reaping,
per-job failure-category breakdown printed at the end of a run — see
`docs/RESILIENCE_FIX_PLAN.md`). Those fixes make the *pipeline* correct. This review looks
at what still makes the *operator's* job harder, specifically for someone babysitting a
run over 1,600+ files across several hours: can they tell what's happening, can they tell
what went wrong, and can a wrong click quietly cost them a re-run.

Findings are ordered by impact. Each has a concrete file/line pointer and an effort
estimate (S = under an hour, M = a focused session, L = a multi-session piece of work).

---

## 1. Every log line looks the same — errors have no visual weight (S)

`gui_components.py`'s `LogPanel` defines an `"error"` text tag styled in `RED`
(`gui_components.py:503`), but nothing in `gui_transcribe.py` ever passes it. Every
message — routine "reviewing…" progress, the `Error: {e}\n{traceback}` dump from a
crashed worker (`gui_transcribe.py:638`), the `_write_summary` failure-category breakdown
that `archive_pipeline.py` now prints at the end of a run, and "Word publication failed
safely for…" — lands in the pane via `stt_log.append(str(msg))` /
`glm_log.append(str(text))` with no tag, i.e. identical plain gray-blue text
(`gui_components.py:494`, `fg="#2c3e50"`).

Concretely: the new `_print_failure_breakdown` (`archive_pipeline.py:3139`) that Fix 7
added is real progress — the operator finally gets "221 failed: 201 path too long…"
instead of nothing. But it's `print()`'d to stdout, which `gui_transcribe.py` redirects
into the same undifferentiated queue as everything else (`_QueueWriter`,
`gui_transcribe.py:413`), so it shows up as a few more lines of identical gray text at
the bottom of a pane that already has thousands of lines in it. An operator scanning by
eye for "did anything go wrong" has nothing to visually latch onto.

**Fix:** in `poll()` (`gui_transcribe.py:793`), tag lines containing `"Error"`,
`"failed"`, `"FAILED"`, `"declined"` as `"error"` before appending; tag lines starting
with a failure-breakdown header (`"failed:"`, `"declined safely:"`) the same way. This is
purely a GUI-layer classification of text already being produced — no pipeline change
needed.

## 2. A stale cutoff date can silently make a whole run a no-op (S/M)

This is the exact "wrong mode silently skips work" risk called out in the brief, and the
code confirms it happens exactly that way. `SettingsPanel.date_var`
(`gui_components.py:294`) defaults to *today* only when no saved value exists —
`ps.get("replace_before_date", datetime.date.today().isoformat())`. Once a project has
been run once with "Refresh transcripts before…" set to some date, `_save_project`
(`gui_transcribe.py:119`) persists that exact date forever; nothing re-freshes it to
"today" on a later session. `_should_process` (`gui_transcribe.py:375`) and the polished
pipeline's equivalent skip logic both compare against that stored date. Six weeks later,
an operator opens the same project, sees the "before" mode still selected (it looks
correct — nothing appears broken), clicks Start, and every file that was touched since
that old date is silently skipped. There's no error, no warning — the run just finishes
fast having done nothing, and the only way to notice is reading the scrolling log
carefully or the end-of-run counts.

**Fix (S):** when loading a project whose saved `replace_mode == "before"`, if the saved
date is more than N days in the past, visually flag the date field (e.g. amber border) or
append a one-line warning: *"This date is 47 days old — files changed since then and
before this date won't be refreshed."*
**Fix (M, better):** before starting a polished-pipeline run, do a cheap dry pass over the
selected folder and show *"1,847 files match; 12 will be (re)processed, 1,835 skipped
under current settings"* in a confirmation step, so the operator sees the effect of their
settings before committing minutes-to-hours of GPU time to it.

## 3. GLM worker count is wrong in the settings panel — visible on every run (S)

`gui_components.py:166` and `gui_components.py:408` both hardcode: *"Local NVIDIA
Parakeet Word → **ten** protected GLM-4.7-Flash review workers → separate GLM Review
Word"*. The actual pipeline runs with `glm_workers=30` (`gui_transcribe.py:288`), and
`gui_transcribe.py` itself gets this right in two other places — the progress message at
`gui_transcribe.py:307` ("**thirty** protected GLM-4.7-Flash workers…") and the pane
label at `gui_transcribe.py:762` ("GLM review queue (**thirty** protected workers)"). The
settings panel — the thing the operator reads *before* clicking Start, to understand what
is about to happen — is the one place still saying "ten." Small, but it's exactly the
kind of inconsistency that erodes trust in every other number the GUI reports during a
multi-hour run.

**Fix:** either hardcode "thirty" in `gui_components.py` to match, or (more robust) pass
the actual worker count into `SettingsPanel` / `update_run_label` so the two files can't
drift again.

## 4. No live progress indicator for the run as a whole — and a working one already exists, unused (M)

For a run that can span 1,600+ files over hours, `launch_gui()` (`gui_transcribe.py:428`)
has no progress bar, percentage, file-count, or ETA anywhere — only two scrolling text
panes. The only way to gauge "how far through are we" is to read and mentally tally
`[index/total]` labels scattered through the STT pane's text, or wait for the final
summary.

`progress_window.py` is a fully built, working answer to exactly this: a compact
always-on-top window with a determinate progress bar, percent label, current-file
display, and batch `[current/total]` counter — thread-safe via its own message queue. It
is **not referenced anywhere else in the codebase** (confirmed: `grep` for
`progress_window`, `init_progress_window`, `TranscriptionProgressWindow` across the repo
returns only `progress_window.py` itself). It appears to be a leftover from an earlier
GUI iteration that was never wired into the current `gui_transcribe.py`.

**Fix (M):** wire `init_progress_window(root)` into `launch_gui()`, and call
`set_file_progress(index, total)` / `set_progress(pct)` from the existing `progress_callback`
plumbing (`self._emit_progress`, `gui_transcribe.py:289`) alongside the current text
append. This closes a real, named gap using code that already exists and works, rather
than building something new.
**If instead the intent is to remove it:** delete it — a maintainer finding an unused,
fully-implemented "progress window" module next to the real GUI will reasonably assume
it's live and waste time chasing why it never appears.

## 5. Exceptions surface as raw tracebacks, not operator messages (S/M)

`worker()`'s catch-all (`gui_transcribe.py:632`) does
`q.put(f"Error: {e}\n{traceback.format_exc()}")` for anything unexpected — including a
`PipelineRunLockError` (`archive_pipeline.py:270`), whose message text is already written
to be operator-readable ("...retry in a moment, or delete {reclaim_mutex} if you..."). But
by the time it reaches the log pane it's buried under a full Python stack trace, in the
same untagged gray text as everything else (see #1). An operator who, say, double-clicks
`run.bat` and starts a second GUI pointed at the same output folder gets a wall of
traceback instead of the one-sentence "another process is already running against this
folder" the exception was designed to say.

**Fix:** in the `except Exception as e:` block, special-case known operator-facing
exception types (`PipelineRunLockError` at minimum) to show just `str(e)` without the
traceback; keep the full traceback for genuinely unexpected exceptions, but tag it as
`"error"` (see #1) so it's at least visually distinct.

## 6. Cancellation gives no feedback once requested (M)

`stop()` (`gui_transcribe.py:645`) sets `STOP_FLAG`, disables the Stop button, and appends
exactly two static lines ("Cancellation requested...") to each pane — then nothing. With
30 GLM workers and one GPU worker each mid-item, cooperative cancellation is checked at
job boundaries (`self.cancel_check()` in `stt_worker`/`cleanup_worker`,
`archive_pipeline.py:2724`/`2764`), so actual shutdown time is unbounded and workload
-dependent. During that gap — which could be seconds or, for a long recording deep in GLM
review, several minutes — the operator has no way to distinguish "still finishing safely"
from "hung." `Run` / `Stop` / `Clear Cache` stay disabled the whole time
(`finish_worker`, `gui_transcribe.py:492`), which is correct, but there's no heartbeat.

**Fix:** on each `_emit_progress` after `STOP_FLAG` is set, or on a simple timer, append a
periodic status line ("Waiting for 3 of 30 GLM workers to reach a safe stopping
point…") so a long cancellation still looks alive rather than frozen.

## 7. Same settings, different meaning depending on pipeline mode (M)

The "Processing selection" radio group — *Skip existing / Refresh all / Refresh
transcripts before…* (`gui_components.py:277`) — uses identical labels regardless of
whether "Polished archive pipeline" is checked. But the comparison behind those labels is
different in each mode: with the pipeline **off**, `_should_process`
(`gui_transcribe.py:375`) compares against the plain sibling `.docx`'s mtime; with the
pipeline **on**, the equivalent decision is made deep inside `archive_pipeline.py` against
per-job manifest/checkpoint state, not a file mtime. An operator who builds a mental model
of "Skip existing" from one mode (e.g. running a quick legacy transcription) and then
switches "Polished archive pipeline" on for the next run carries that mental model into a
system with different semantics, with no UI cue that anything changed.

**Fix:** either genuinely unify the semantics, or add a short mode-specific note under the
radio group (the panel already does this well elsewhere — see
`existing_transcripts_note`, `gui_components.py:373`) explaining what "existing" means in
the current mode.

## 8. Two decoupled helper processes launch with no lifecycle visibility (S)

"Context Finder" (`_launch_context_finder_process`, `gui_transcribe.py:191`) and "Cleanup
Access" (`configure_cleanup_access`, `gui_transcribe.py:687`) both fire-and-forget via
`subprocess.Popen`, catching only the exception from *starting* the process
(`gui_transcribe.py:699`). If the child window fails after launch — e.g. a second click
spawns a duplicate Context Finder window, or the child crashes once its own GUI is up —
there is no feedback in the main window at all. This is minor on its own, but it's the
same "silent failure" shape as items #2 and #5: the GUI's default posture is to say
nothing rather than something went wrong.

**Fix:** track the `Popen` handle and disable/relabel the launching button while the
child is alive (`proc.poll() is None`), the same pattern already used for `run_btn`
during a pipeline run.

## 9. `run.bat` swallows the actual reason both venvs failed (S)

`run.bat:13` and `run.bat:16` both redirect the venv health-check's stderr to `nul`
(`2>nul`), so when neither `.venv` nor `.parakeet-venv` has a working
`keyring`/`psutil`/`tkinter`/`docx`, the operator sees only "AudioProcessor's local
environments are incomplete" with no indication of *which* import failed. That's the
difference between a two-minute `pip install` fix and reaching for
`install_geforce.ps1 -RecreateVenv` (a much bigger hammer) on a guess.

**Fix:** drop the `2>nul` on at least the second (fallback) check, or capture and echo the
Python error text before the generic message.

## 10. Unbounded log growth and no way to search 2,000 files' worth of output (M)

`LogPanel.append` (`gui_components.py:508`) inserts into a `tk.Text` with no cap and
always calls `.see("end")`. Over a full archive run this is tens of thousands of lines
across two widgets, growing memory and eventually scroll/redraw cost with no ceiling, and
there is no way to filter the pane to just the failures — the operator's only tool is
eyeballing a wall of interleaved 30-worker output (each GLM line prefixed
`"GLM worker N: ..."`, but otherwise unsorted by worker) or waiting for the final summary.
Also worth noting: because `.see("end")` runs on every append, a user who manually
scrolls up to review an earlier error gets yanked back to the bottom on the very next
message — there's no "pause autoscroll while the user has scrolled up" behavior.

**Fix (S) for the scroll-yank:** only call `.see("end")` when the view is already at (or
near) the bottom before the insert.
**Fix (M) for the rest:** trim old lines past some count (e.g. keep last 5,000), and add a
simple "failures only" filter toggle that re-renders from an in-memory list of
`(tag, text)` tuples already being appended.

---

## Summary table

| # | Issue | Effort |
|---|---|---|
| 1 | No color/tag distinction for errors in either log pane | S |
| 2 | Stale "Refresh before" date can silently no-op a whole run | S/M |
| 3 | Settings panel says "ten" GLM workers; actual is thirty | S |
| 4 | No live progress bar/ETA; a working one (`progress_window.py`) exists unused | M |
| 5 | Exceptions (incl. run-lock conflicts) show as raw tracebacks | S/M |
| 6 | No feedback during cancellation — looks identical to "hung" | M |
| 7 | "Processing selection" labels mean different things in legacy vs. polished mode | M |
| 8 | Context Finder / Cleanup Access child processes have no lifecycle tracking | S |
| 9 | `run.bat` hides the actual venv import error behind `2>nul` | S |
| 10 | Unbounded log growth, scroll-yank on autoscroll, no failure filter | S/M |

None of these require touching the resilience work already landed this week — they're
all in the GUI/operator-communication layer sitting on top of it, and #1 and #4 in
particular are the highest-leverage: a working failure-category breakdown and a working
progress-bar widget both already exist in the codebase and just aren't connected to what
the operator actually sees.
