# LifeMate Docker handoff

The uploaded judge guide (`Docker_Guide_ZAC2026_LifeMate.pdf`, pp. 5–18) requires the frozen inference image, the shared timing notebook, the training reproduction material, and a Google Drive link plus the MD5 of the exact `.tar.gz`. The leaderboard JSONL is a separate submission.

The build uses Molab for the merged weights and Docker Desktop's local Linux daemon for the image layers. The laptop pulls only listed model files over the existing Molab file endpoint and verifies SHA-256 and size. Docker stays on its local Windows named pipe. A Docker CLI running inside Molab would require a separately reachable authenticated Docker endpoint. This build uses the verified file transfer to avoid that dependency. It performs a CPU package smoke check and does not use the laptop GPU.

## Step-548 fallback

The active build logs are in `results/judge_qwen35/step548-build.log` and `results/judge_qwen35/step548-build.stderr.log`. Resume from the repository root if the build stops. Supply the active runtime URL if it changes.

```powershell
& .\judge_qwen35\build_from_molab.ps1 -RuntimeUrl 'https://sb-d9d3a5011a88b1be.sb.molab.run' -Candidate step548
```

`DOCKER_ARCHIVE_READY` prints the archive path and its MD5 after `docker load` succeeds. The default output is `results/judge_qwen35/step548/zac2026_LifeMate.tar.gz` with an adjacent `.md5`. Set `-Team` to the registered team's archive name before export, or rename the completed archive without altering its bytes.

## Full-TRAIN candidate

Wait for FULL_TRAIN_PUBLIC_READY and compare its public score with step 548. If the full-TRAIN candidate wins, download training/judge-models.json from the private Hugging Face backup under qwen35-full-train/from-step548/. Pass it to the build launcher with -Manifest PATH -Candidate full-train. The manifest identifies every merged weight and tokenizer file and verifies its size and SHA-256. Its paths refer to the Molab runtime that built the models. If that runtime expires, restore the pinned base and backed-up adapter, remerge, and verify the files before building.

## Judge handoff

The build uses the CUDA 13.2 base from the guide and the tested Molab vLLM and PyTorch versions. Start the script with the judge's read-only mount at `/data` and result mount at `/result`. Run the same mounted sample through the timing notebook and compare IDs, tool calls, and final state hashes. The `time` field may vary. The first step-150 judge test stopped during vLLM startup. The new package prints the vLLM log tail into the judge-visible container log if that happens again.

GPU inference, timing on the judge hardware, and notebook output parity require a GPU Docker test. CPU packaging checks do not exercise model startup on the RTX 5090. Upload the finished archive to Google Drive, enable the viewer link required by the judge, confirm it is downloadable, then paste the link and the 32-character MD5 into the portal. Keep the exact artifact used for the test.
