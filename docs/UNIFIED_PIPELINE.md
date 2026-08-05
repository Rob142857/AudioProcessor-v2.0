# Unified archival transcription pipeline

Status: bridge implementation complete; live one-file canary pending a tested Python environment and a dedicated Cloudflare Access service token.

## What is now wired together

The local tool remains responsible for reading private audio and producing the first transcription. The protected research application remains responsible for glossary-grounded GLM cleanup. `archive_pipeline.py` joins them into one resumable process:

```text
recording
  -> local tape preprocessing
  -> Faster-Whisper large-v3
  -> immutable raw text + segment JSON + VTT + SRT
  -> pinned glossary snapshot
  -> protected GLM-4.7-Flash cleanup
  -> fidelity/coverage checks
  -> final DOCX + manifest + status report
```

The source archive and its existing DOCX files are never overwritten. By default, generated artifacts go into a separate sibling folder named `<archive> - Polished`.

Every recording receives a collision-proof directory containing:

```text
<relative folders>/<recording stem>__<source extension>/
  manifest.json
  run.jsonl
  raw.txt
  stt.formatted.txt (only when distinct from the model output)
  raw.segments.json
  raw.vtt
  raw.srt
  cleanup-chunks/
  cleanup.json
  cleaned.txt
  qa.json
  final.docx
```

The output root also contains `pipeline.sqlite3`, `pipeline-status.csv`, and `last-run-summary.json`. A restart reuses every valid completed stage. It does not infer completion merely from the presence of a DOCX.

## One-click use

Use a dedicated Cloudflare Access service token whose policy is limited to the transcript-cleanup application. Configure it once; the values are validated and stored in Windows Credential Manager:

```powershell
.venv\Scripts\python.exe configure_cleanup_credentials.py
./run_full_pipeline.bat
```

For an unattended process, `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` may instead be supplied together in that process's environment; they override the credential store. Use `configure_cleanup_credentials.py --clear` to remove the saved pair.

With no arguments, the runner opens a folder chooser. For a safe first canary:

```powershell
./run_full_pipeline.bat "C:\path\to\sample recordings" --limit 1
```

Useful controls:

- `--dry-run` inventories supported recordings without transcribing them.
- `--limit N` processes only the first N recordings.
- `--no-cleanup` produces immutable local STT artifacts without calling Cloudflare.
- `--cleanup-only` reuses raw artifacts and retries cleanup/rendering.
- `--render-only` rebuilds DOCX from existing cleaned text.
- `--retry-review` retries only work which previously reached `needs_review`.
- `--force` deliberately reruns all selected stages.
- Ctrl+C stops between files; completed checkpoints remain reusable.

The service token is never written to settings, logs, manifests, or checkpoint files. The client refuses insecure non-local HTTP endpoints and Access login redirects/HTML, and validates access once before creating per-recording work.

## Archive findings

The read-only archive inventory on 5 August 2026 found:

- 2,278 supported recordings after adding AIFF, AIF, and 3GP discovery.
- 2,260 recordings recognised by the old extension list.
- 6 AIFF recordings that the old recursive runner silently omitted.
- 9 locations where two audio formats have the same stem and therefore target the same legacy DOCX path.
- 2,259 existing DOCX files, which are intentionally preserved.

The dry-run completed against the real archive and found all 2,278 recordings without modifying the archive.

## Model choice

There is no defensible universal winner for deteriorated tapes without a representative gold-set comparison. The production default therefore remains local Faster-Whisper `large-v3`: it is the strongest known control on this collection, fits the 8 GB GTX 1070 Ti using CTranslate2 INT8, and keeps audio local.

Recommended challengers are:

| Candidate | Best use | Important trade-off |
| --- | --- | --- |
| Local Faster-Whisper `large-v3` | Fidelity control and private bulk processing | Existing host is Pascal; use a tested Python/CUDA/CTranslate2 environment |
| Cloudflare Whisper `large-v3-turbo` | Cheap overflow and speed comparison | Approximately $3.06 per 100 audio hours, but turbo is not automatically more accurate than full large-v3 |
| Cloudflare Deepgram Nova-3 | Australian English, word confidence, keyterms, overlapping Q&A, and diarization | Approximately $31.20 per 100 audio hours; set `language=en-AU` and `mip_opt_out=true` |
| NVIDIA Canary-1b-v2 | Local noisy-speech accuracy challenger | Best trialled in WSL2/Linux; not yet a proven Windows/Pascal replacement |
| Faster-Whisper plus pyannote Community-1 | Speaker attribution while retaining the known ASR | Run ASR and diarization sequentially on 8 GB VRAM |

Parakeet TDT and VibeVoice are interesting, but neither is the right default on this computer: NVIDIA's Parakeet model card lists Volta or newer as supported, while the available VibeVoice 8B checkpoint is too large for an 8 GB GTX 1070 Ti.

Current official references:

- [Faster-Whisper requirements and batched inference](https://github.com/SYSTRAN/faster-whisper)
- [Cloudflare Whisper large-v3-turbo](https://developers.cloudflare.com/workers-ai/models/whisper-large-v3-turbo/)
- [Cloudflare Nova-3](https://developers.cloudflare.com/workers-ai/models/nova-3/)
- [Cloudflare Workers AI pricing](https://developers.cloudflare.com/workers-ai/platform/pricing/)
- [Cloudflare Workers AI data usage](https://developers.cloudflare.com/workers-ai/platform/data-usage/)
- [NVIDIA Canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2)
- [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

## Required model bake-off

Before starting all 2,278 recordings, hand-correct 60–120 minutes spread across 10–15 excerpts:

- clean lecturer speech;
- hiss, hum, low level, clipping, reverberation, and dropout;
- long silence, music, and tape-side changes;
- terminology-heavy passages;
- Australian accents;
- audience questions and overlapping speakers.

Measure raw word error rate, proper-name/technical-term recall, deletions, silence hallucinations, duration coverage, timestamp error, real-time factor, failures/retries, and archive-hour cost. Score raw ASR before GLM cleanup so the editor cannot conceal an ASR regression.

Copy `benchmark-manifest.example.json`, point it at hand-corrected references and each model's raw transcript, then score the candidates:

```powershell
.venv\Scripts\python.exe benchmark_transcripts.py benchmark-manifest.json --output benchmark-results
```

The command writes both JSON detail and a CSV model summary. Do not promote a challenger unless it improves the terminology-heavy/noisy subsets without a material deletion or hallucination regression.

## Why this is a bridge rather than the final server architecture

The current `/api/tooling/cleanup-chunk` endpoint is synchronous and has no server-side job idempotency. The local runner compensates with immutable raw artifacts, a pinned snapshot of the editable glossary overlay, an explicitly pinned GLM model, sequential checkpoints, retries, and a manifest. The server merges that overlay with its curated built-in baseline, so the whole merged glossary remains eligible for each chunk. The current endpoint does not expose a version hash for the built-in baseline, and a lost HTTP response can still repeat paid inference; both limitations are addressed by the proposed v2 job API.

The next server uplift should replace browser-owned cleanup progress with a Cloudflare Workflow:

1. Upload raw transcript to a dedicated cleanup R2 bucket by content hash.
2. Create an idempotent job with the GLM model, pipeline version, and immutable glossary version pinned once.
3. Store only job/chunk metadata in a dedicated cleanup D1 database.
4. Let a Workflow perform sequential context-linked cleanup with durable retries.
5. Expose `queued`, `running`, `needs_review`, `completed`, `approved`, and `published` states.
6. Publish approved transcript bodies directly into search ingestion with source-audio and cleanup hashes.

Queues are optional later for burst backpressure. Queue messages should contain job IDs, never transcripts.

That migration requires new Cloudflare resources and an explicit publish/approval policy, so it should be a separate controlled deployment rather than an implicit side effect of this local-tool change.

## Known environment gate on this machine

The clone currently resolves to Python 3.13.3 with no local `.venv` and no installed PyTorch, Faster-Whisper, OpenAI Whisper, python-docx, or psutil packages. The bundled FFmpeg works and the NVIDIA GTX 1070 Ti is visible.

Use Python 3.11 or 3.12 for the first reproducible environment. The current Faster-Whisper documentation requires CUDA 12 cuBLAS and cuDNN 9 for its newest CTranslate2 releases; the CUDA version displayed by `nvidia-smi` is a driver capability, not proof that the needed runtime DLLs are installed. Do not begin the archive-wide job until `pipeline_doctor.py` passes and a one-file canary has been reviewed.
