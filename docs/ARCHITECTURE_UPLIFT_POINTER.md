# Architecture Uplift Proposal — Pointer

A whole-ecosystem architecture and duplication-cleanup proposal covering this repo, searchagent, and pg-access-worker was written and critiqued on 2026-08-14.

Full document: `searchagent/docs/ARCHITECTURE_UPLIFT_PROPOSAL.md` (commit c95e54a, branch codex/research-grounding-audit). Status: proposal only, no code changed — read it before acting on any item below.

Five items in it touch this repo (AudioProcessor-v2.0) directly:

1. Delete `archive_deletewhenready/` — P0, zero-risk, 56 files / 15 MB.
2. `progress_window.py` is explicitly spared from deletion; the proposal defers to `docs/UI_UX_ARCHITECTURE_REVIEW_2026-08-09.md`'s recommendation to wire it into `launch_gui()` instead (Decision #10).
3. Fix the GLM worker-count label in `gui_components.py` ("ten" vs. the actual thirty) — P0 quick win.
4. Extract `fsutil.py`; fold `archive_older_transcripts.py`, `reset_corrupted_transcripts.py`, `legacy_docx_replace.py`, and `prepare_docx_cleanup.py` into one `archive_doctor.py` CLI (P1/P2).
5. Unify `PipelineConfig` into one schema (P1), and have `archive_pipeline.py` start emitting `manifest.jsonl` — the new document-identity contract with searchagent, replacing the filename-suffix convention (P1, §2.2).

Context Finder (`context_finder_client.py`, `context_compilation_inventory.py`) is also named in Decision #8, but no code change is proposed for it yet.
