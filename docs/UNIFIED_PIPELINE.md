# Unified archival transcription pipeline

Status: bridge implementation complete. The pinned Python/CUDA environment and a real 81.7-second tape canary have passed on the GTX 1070 Ti using Faster-Whisper tiny with CUDA/INT8. Review a representative large-v3 canary before the unrestricted archive run.

## What is now wired together

The local tool remains responsible for reading private audio and producing the first transcription. The protected research application remains responsible for glossary-grounded GLM cleanup. `archive_pipeline.py` joins them into one resumable process:

```text
recording
  -> local tape preprocessing
  -> Faster-Whisper large-v3 + token-budgeted pinned-glossary hotwords
  -> immutable raw text + segment JSON + VTT + SRT
  -> pinned glossary snapshot
  -> protected GLM-4.7-Flash cleanup
  -> fidelity/coverage checks
  -> raw Whisper DOCX + GLM Review DOCX + manifest + status report
```

For recordings whose legacy Faster-Whisper DOCX already exists, the same runner
also has a distinct raw-input route:

```text
source-adjacent legacy DOCX
  -> conservative Transcript: section import (audio and Whisper skipped)
  -> immutable raw text + DOCX container/text hashes
  -> pinned glossary snapshot
  -> protected GLM-4.7-Flash cleanup
  -> imported-text/hash checks (timestamp coverage is explicitly not applicable)
  -> separate GLM Review DOCX + manifest + status report
```

This is not `cleanup-only` and it does not fabricate empty Whisper segments.
No segment JSON, VTT, SRT, audio duration, or coverage claim is created. The
legacy importer fails closed on ambiguous document structure and keeps the body
only in hashed `raw.txt`, never in the manifest.

Generated artifacts always go into a separate output folder. A folder batch defaults to the sibling `<archive> - Polished` directory. Publication never writes metadata sidecars into recording folders. Fresh STT may create or refresh the raw `<stem>.docx` selected by policy, while GLM output always uses `<stem> - GLM Review.docx`. Imported-DOCX mode leaves its source Word file byte-for-byte unchanged.

Every recording receives a collision-proof directory containing:

```text
<relative folders>/<recording stem>__<source extension>/
  manifest.json
  run.jsonl (optional troubleshooting log)
  raw.txt
  stt.formatted.txt (only when distinct from the model output)
  raw.segments.json
  raw.vtt
  raw.srt
  cleanup-chunks/
  cleanup.json
  cleaned.txt
  qa.json
  whisper.docx (fresh STT only; pre-GLM)
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

The desktop GUI is launched with `run.bat`. Its **Polished archive pipeline**
switch is on by default and shows the fixed route: local Faster-Whisper Word →
protected GLM-4.7-Flash cleanup → separate GLM Review Word. It uses the same manifests,
checksums, cleanup credentials, separate polished output root, verification,
backup, and guarded source-DOCX publisher as the command-line runner.
Folder runs use the stable sibling `<archive> - Polished` output. A GUI
single-file canary automatically uses its own dedicated directory under the
sibling `<source-folder> - Polished Single Files` tree, keeping generated work
outside the guarded source scope and making a restart unambiguous.

Select **Use existing Word transcripts (skip Whisper)** for an archive that
already contains source-adjacent legacy DOCX files. Choose **Refresh all** or
**Refresh transcripts before…**; **Skip existing** is deliberately invalid in
this mode because every raw input is an existing document. Whisper model and
quality controls are disabled, the preflight needs no GPU/audio stack, and the
flow label changes to existing Word → protected GLM-4.7-Flash → separate GLM Review Word.
The advanced Force control reruns GLM and rendering from the hash-verified
preserved import; it never causes a DOCX to be re-imported.

The three existing-output policies select recordings by the source-adjacent
DOCX: missing only, all, or missing plus documents whose local modification
time is before a strict `YYYY-MM-DD` date. Invalid dates stop before audio is
opened. This selection is independent of the advanced **Reprocess from audio**
checkbox: with that checkbox off, an interrupted run reuses every valid raw,
cleanup, and render checkpoint. The GUI Stop button requests cancellation from
both the pipeline and active Whisper engine. Completed recordings, completed
stages, and completed GLM chunks resume; Whisper does not checkpoint partial
decoding within one recording, so the current recording restarts from its
beginning if it is stopped during Whisper. Each completed review copy is
published atomically before the runner advances, so a later cancellation keeps
finished siblings while the current incomplete job remains resumable.
Closing the GUI while work is active uses the same safe-stop path and keeps the
window alive until the worker and any atomic Word publication transaction have
finished; it never abandons the publisher merely because the close button was
pressed.

Before local transcription starts, a publication run also groups the already
selected recordings by their eventual source-adjacent ` - GLM Review.docx` target. If two
formats or files would map to the same Word filename, the run stops immediately
and lists every conflicting source path. The tool never guesses which recording
is authoritative; select one source from each reported group before retrying.

Skip-Whisper mode instead discovers the unique DOCX itself. Same-stem audio
formats collapse onto that one document without choosing or reading an audio
file; every candidate recording path is retained as provenance in the manifest.

With no arguments, the runner opens a folder chooser. For a safe first canary:

```powershell
./run_full_pipeline.bat "C:\path\to\sample recordings" --limit 1 --no-publish-source-docx
```

Useful controls:

- `--dry-run` inventories supported recordings without transcribing them.
- `--limit N` processes only the first N recordings.
- `--no-cleanup` produces immutable local STT artifacts without calling Cloudflare.
- `--cleanup-only` reuses raw artifacts and retries cleanup/rendering.
- `--render-only` rebuilds DOCX from existing cleaned text.
- `--retry-review` retries only work which previously reached `needs_review`.
- `--force` deliberately reruns all selected stages.
- `--existing-transcripts-only` (aliases `--use-existing-docx` and `--skip-stt`) imports legacy DOCX bodies and runs GLM/render only; combine it with `--existing-docx-mode all` or `before`.
- `--no-troubleshooting-logs` disables optional per-job event logs and full terminology-list duplication; compact hash/provenance checkpoints remain.
- `--no-publish-source-docx` is a launcher-only opt-out which leaves final DOCX files solely in the polished output tree.
- Ctrl+C stops between files; completed checkpoints remain reusable.

The batch launcher enables UTF-8 console/Python I/O, selects the doctor mode which matches the requested work, and requires CUDA for any run that transcribes audio. Cleanup-only and render-only runs do not require the GPU; dry runs use the inventory-only doctor mode.

### Safe source-adjacent review publication

The launcher adds `--publish-source-docx` by default. Publication occurs per completed recording. For fresh STT it publishes the pre-GLM Word transcript as `<stem>.docx` and the cleaned document as `<stem> - GLM Review.docx`. Imported-DOCX mode publishes only the review sibling and requires the source DOCX container hash to remain exactly unchanged.

Single-file publication is intentionally supported for a reviewed canary. Its guarded source scope is the input file's parent directory, so always pass a new, dedicated `--output` directory which contains only that canary's generated manifest and artifacts. Do not reuse a shared `Polished Transcripts` directory for this purpose. Folder batches remain the normal archive workflow.

```powershell
.\run_full_pipeline.bat "C:\path\to\recordings\one-tape.wav" --output "C:\path\to\fresh-canary-output"
```

Both `verified` and `needs_review` final jobs may produce a review copy because the document is explicitly awaiting a human word-for-word check. QA status and approval are not conflated: every review manifest, publication record and transaction report says `approval_state: pending_human_review`. Failed, cancelled and incomplete jobs do not publish. Dry runs and limited runs remain non-publishing.

Each GLM Review document ends with a restrained provenance note in this form: `Processed by speech-to-text from a digitised tape recording originally recorded in person by MW on 22 January 1985.` A second removable line reads `Needs human review.` until a person has completed the word-for-word check. Raw Whisper documents do not carry the GLM review notice. Old cleanup timestamps, model/device details, and earlier generated provenance text are removed during rendering rather than appearing in the publication.

For fresh STT, transcript completeness is a hard verification and publication gate. There must be at least one text-bearing STT segment with a valid end timestamp, the source audio duration must be known, and the final segment must reach the end within the greater of 2 seconds or 5% of the recording duration, capped at 120 seconds. This is the documented trailing-silence tolerance; a longer gap is sent to review instead of being assumed silent. Publication re-reads the hashed segment artifact and repeats this check. Imported-DOCX mode instead rechecks the exact DOCX container → extracted raw text → GLM output → render hash chain and records STT coverage as `not_applicable`; it never claims timestamp completeness it cannot prove. Both routes require recorded glossary-grounding minimum and maximum counts to be integers at least as large as the pinned glossary count.

Before changing an older tool-generated review (or a deliberately refreshed fresh-STT Word file), the publisher copies and verifies it inside the separate polished workspace:

```text
publication-backups/<UTC run-id>/<relative source folders>/
```

Each completed job is staged and committed atomically. A mid-commit failure rolls back changed targets; verified backups are retained after success or failure. An existing review target is replaceable only when its current hash is proven by an immutable prior journal. A manually edited review copy fails closed. The source recording folders contain only the two intended Word documents, never logs, manifests, backups or checkpoint sidecars.

Before commit, the publisher also writes a unique per-run planned journal. If
the process or machine stops after an atomic replacement but before the final
published receipt, the next run accepts that partial commit only after it
rechecks the journal's plan hash, target mapping and current generated hash; a
replacement additionally requires its byte-exact original backup. It can then
finish safely without ever importing a ` - GLM Review.docx` as source text.

The service token is never written to settings, logs, manifests, or checkpoint files. The client refuses insecure non-local HTTP endpoints and Access login redirects/HTML, and validates access once before creating per-recording work.

## Archive findings

The read-only archive inventory on 5 August 2026 found:

- 2,278 supported recordings after adding AIFF, AIF, and 3GP discovery.
- 2,260 recordings recognised by the old extension list.
- 6 AIFF recordings that the old recursive runner silently omitted.
- 22 same-stem multi-format groups; fresh STT treats these as collisions, while skip-Whisper mode cleans each shared DOCX once without selecting an audio variant.
- 2,259 existing DOCX files. Skip-Whisper publication preserves every source transcript and creates separate ` - GLM Review.docx` siblings for human checking.
- 2,250 canonical source-adjacent DOCX inputs selected by skip-Whisper discovery; six unique recording names have no adjacent DOCX and nine alternate/old DOCX names do not exactly match a recording stem.

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

## Validated environment on this machine

The supported lane is Python 3.12 x64, PyTorch 2.6.0+cu124, Faster-Whisper 1.2.1, and CTranslate2 4.8.1. The doctor has verified CUDA 12.4, the GTX 1070 Ti's compute capability 6.1, `sm_61` wheel support, pinned cuBLAS/cuDNN loading, one CTranslate2 CUDA device, and Pascal-safe INT8/INT8-Float32. `nvidia-smi` may display CUDA 13.0 because that is the driver's maximum API capability; it is not the runtime selected by this environment.

Run the reviewed local `install_geforce.ps1`. It first tries `py -3.12`, then `%LOCALAPPDATA%\Programs\Python\Python312\python.exe`, and creates `.venv` with whichever exact Python 3.12 x64 interpreter passes its probe. Do not begin the archive-wide job until `pipeline_doctor.py --mode full --require-gpu` passes and a representative real-tape large-v3 canary has been reviewed.
