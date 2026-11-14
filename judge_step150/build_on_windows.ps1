param(
    [switch]$SkipWeightDownload,
    [string]$Image = 'zac2026-lifemate:qwen35-step150'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Push-Location $repo
try {
    $server = docker version --format '{{.Server.Os}}/{{.Server.Arch}}'
    if ($LASTEXITCODE -or $server -ne 'linux/amd64') {
        throw 'Start Docker Desktop in Linux containers mode, then rerun.'
    }
    if (!$SkipWeightDownload) { & "$PSScriptRoot/stage_models_from_molab.ps1" }
    foreach ($model in @('agent', 'asr')) {
        if (!(Test-Path "$PSScriptRoot/staged_models/$model/model.safetensors")) {
            throw "Missing staged $model weights."
        }
    }

    docker build --platform linux/amd64 -f judge_step150/Dockerfile -t $Image .
    if ($LASTEXITCODE) { throw 'Docker build failed.' }
    docker run --rm --network none --entrypoint bash $Image -ec 'cd /code; test -s saved_models/agent/model.safetensors; test -s saved_models/asr/model.safetensors; test -s predict_notebook.ipynb; python judge_predict.py --help'
    if ($LASTEXITCODE) { throw 'Offline CPU package check failed.' }

    $dir = Join-Path $repo 'results/lifemate_qwen35_150/judge'
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $tar = Join-Path $dir 'zac2026_lifemate_qwen35_step150.tar'
    $archive = "$tar.gz"
    if ((Test-Path $tar) -or (Test-Path $archive) -or (Test-Path "$archive.partial")) {
        throw "Archive already exists in $dir. Move it before rerunning."
    }
    docker save --output $tar $Image
    if ($LASTEXITCODE) { throw 'Docker save failed.' }
    python -c "import gzip,shutil,sys; a=open(sys.argv[1],'rb'); b=gzip.open(sys.argv[2],'wb',compresslevel=3); shutil.copyfileobj(a,b,length=8*1024*1024); b.close(); a.close()" $tar "$archive.partial"
    if ($LASTEXITCODE) { throw 'Archive compression failed.' }
    Move-Item -LiteralPath "$archive.partial" -Destination $archive
    docker load --input $archive
    if ($LASTEXITCODE) { throw 'Docker archive reload failed.' }
    $md5 = (Get-FileHash -Algorithm MD5 -LiteralPath $archive).Hash.ToLowerInvariant()
    $md5 | Set-Content -Encoding ascii -LiteralPath "$archive.md5"
    Write-Host "ARCHIVE $archive"
    Write-Host "MD5 $md5"
} finally {
    Pop-Location
}
