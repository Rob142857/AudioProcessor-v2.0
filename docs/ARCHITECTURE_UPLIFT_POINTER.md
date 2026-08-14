# Architecture Uplift Proposal — Pointer

A whole-ecosystem architecture and duplication-cleanup proposal covering this repo, searchagent, and pg-access-worker was written and critiqued on 2026-08-14.

Full document: `searchagent/docs/ARCHITECTURE_UPLIFT_PROPOSAL.md` (branch
`codex/research-grounding-audit`). The uplift was implemented on 2026-08-14;
its delivery record is
`searchagent/docs/ARCHITECTURE_UPLIFT_IMPLEMENTATION.md`.

Five items in it touch this repo (AudioProcessor-v2.0) directly:

1. `archive_deletewhenready/` removed — 116 tracked files, about 15 MB,
   recoverable from Git history.
2. `progress_window.py` is explicitly spared from deletion; the proposal defers to `docs/UI_UX_ARCHITECTURE_REVIEW_2026-08-09.md`'s recommendation to wire it into `launch_gui()` instead (Decision #10).
3. Fix the GLM worker-count label in `gui_components.py` ("ten" vs. the actual thirty) — P0 quick win.
4. Extract `fsutil.py`; fold `archive_older_transcripts.py`, `reset_corrupted_transcripts.py`, `legacy_docx_replace.py`, and `prepare_docx_cleanup.py` into one `archive_doctor.py` CLI (P1/P2).
5. `PipelineConfig` now resolves through one schema, and `archive_pipeline.py`
   emits `.transcription-manifest/publications.jsonl` as the new explicit
   document-identity contract with searchagent. Filename matching remains only
   as a warned one-cycle compatibility fallback.

Context Finder (`context_finder_client.py`, `context_compilation_inventory.py`) is also named in Decision #8, but no code change is proposed for it yet.
