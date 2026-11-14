param(
    [Parameter(Mandatory=$true)][uri]$RuntimeUrl,
    [string]$Manifest,
    [ValidatePattern('^[a-z0-9][a-z0-9-]*$')][string]$Candidate = 'step548',
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$Team = 'LifeMate',
    [string]$OutputRoot,
    [switch]$BuildOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
if (!$Manifest) { $Manifest = Join-Path $PSScriptRoot 'step548-models.json' }
if (!$OutputRoot) { $OutputRoot = Join-Path $repo 'results/judge_qwen35' }
$job = Join-Path $OutputRoot $Candidate
$weights = Join-Path $job 'weights'
New-Item -ItemType Directory -Force -Path $weights | Out-Null
$modelManifest = Get-Content -LiteralPath $Manifest -Raw | ConvertFrom-Json
$endpoint = $RuntimeUrl.AbsoluteUri.TrimEnd('/') + '/api/files/download'
foreach ($name in @('agent','asr')) {
    $group = $modelManifest.$name
    if (!$group.files.'model.safetensors') { throw "Missing $name weights manifest" }
    $folder = Join-Path $weights $name
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    foreach ($entry in $group.files.PSObject.Properties) {
        $file = $entry.Name
        if ($file -notmatch '^[A-Za-z0-9_.-]+$' -or $file -in @('.','..')) { throw "Invalid model entry: $file" }
        $target = Join-Path $folder $file
        if (!(Test-Path -LiteralPath $target)) {
            $partial = "$target.partial"
            if (!(Test-Path -LiteralPath $partial) -or (Get-Item -LiteralPath $partial).Length -ne $entry.Value.size) {
                Write-Host "FETCH $name/$file"
                curl.exe --fail --show-error --retry 3 --connect-timeout 15 --continue-at - --get --data-urlencode "path=$($group.path)/$file" --output $partial $endpoint
                if ($LASTEXITCODE) { throw "Download stopped for $name/$file. Rerun to resume." }
            }
            if ((Get-Item -LiteralPath $partial).Length -ne $entry.Value.size -or (Get-FileHash -LiteralPath $partial).Hash -ne $entry.Value.sha256) {
                throw "Verification failed for $partial. Inspect or move the partial file before retrying."
            }
            Move-Item -LiteralPath $partial -Destination $target
        } elseif ((Get-Item -LiteralPath $target).Length -ne $entry.Value.size -or (Get-FileHash -LiteralPath $target).Hash -ne $entry.Value.sha256) {
            throw "Existing artifact failed verification: $target"
        }
    }
}
$release = [ordered]@{candidate=$Candidate; agent=$modelManifest.agent.files; asr=$modelManifest.asr.files}
if ($modelManifest.run) { $release.run = $modelManifest.run }
[IO.File]::WriteAllText((Join-Path $weights 'release.json'),($release | ConvertTo-Json -Depth 8)+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
$image = "zac2026-lifemate:qwen35-$Candidate"
docker --context desktop-linux buildx build --load --platform linux/amd64 --progress plain --build-context "weights=$weights" -f (Join-Path $PSScriptRoot 'Dockerfile') -t $image $repo
if ($LASTEXITCODE) { throw 'Docker build failed' }
docker --context desktop-linux run --rm --network none --entrypoint bash $image -ec 'test -s /code/release.json; test -s /code/saved_models/agent/model.safetensors; test -s /code/saved_models/asr/model.safetensors; python /code/judge_predict.py --help; jupyter lab --version'
if ($LASTEXITCODE) { throw 'Offline CPU package check failed' }
Write-Host "DOCKER_IMAGE_READY $image"
if ($BuildOnly) { return }
$archive = Join-Path $job "zac2026_${Team}.tar.gz"
if (Test-Path -LiteralPath $archive) { throw "Archive already exists: $archive" }
python (Join-Path $PSScriptRoot 'save_archive.py') $image $archive
if ($LASTEXITCODE) { throw 'Docker save or gzip failed' }
docker --context desktop-linux load --input $archive
if ($LASTEXITCODE) { throw 'Docker archive reload failed' }
$md5 = (Get-FileHash -Algorithm MD5 -LiteralPath $archive).Hash.ToLowerInvariant()
$md5 | Set-Content -Encoding ascii -LiteralPath "$archive.md5"
Write-Host "DOCKER_ARCHIVE_READY $archive $md5"
