param(
    [string]$RuntimeUrl = 'https://sb-a70daef07ff59a21.sb.molab.run'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $PSScriptRoot 'staged_models'
$remoteRoot = '/marimo/results/lifemate-qwen35-recovery'
$agentFiles = @('config.json','generation_config.json','model.safetensors',
    'preprocessor_config.json','chat_template.jinja','video_preprocessor_config.json',
    'tokenizer_config.json','merges.txt','vocab.json','tokenizer.json')
$asrFiles = @('added_tokens.json','config.json','generation_config.json','merges.txt',
    'model.safetensors','normalizer.json','preprocessor_config.json',
    'special_tokens_map.json','tokenizer.json','tokenizer_config.json','vocab.json')
$groups = @(
    @{ Name='agent'; Path="$remoteRoot/continuation-150/merged"; Files=$agentFiles;
       Hash='c69e42825c46e0571c09ce0468cf267f2f573602e5e3a5f90406ceea0881edaa' },
    @{ Name='asr'; Path="$remoteRoot/pho-merged"; Files=$asrFiles;
       Hash='c2bc73b7300c76ecc4736b4b48d5d08344a4a22c046d0a0b4ca8ad67f052f2f2' }
)
$endpoint = $RuntimeUrl.TrimEnd('/') + '/api/files/download'
foreach ($group in $groups) {
    $folder = Join-Path $destination $group.Name
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    foreach ($file in $group.Files) {
        $target = Join-Path $folder $file
        if (Test-Path -LiteralPath $target) {
            if ($file -eq 'model.safetensors' -and
                (Get-FileHash -LiteralPath $target).Hash -ne $group.Hash) {
                throw "Existing model has an unexpected checksum: $target"
            }
            continue
        }
        $partial = "$target.partial"
        Write-Host "Fetching $($group.Name)/$file"
        curl.exe --fail --show-error --progress-bar --retry 3 --connect-timeout 15 `
            --continue-at - --get --data-urlencode "path=$($group.Path)/$file" `
            --output $partial $endpoint
        if ($LASTEXITCODE) { throw "Transfer stopped for $file. Rerun this script to resume." }
        if ($file -eq 'model.safetensors' -and
            (Get-FileHash -LiteralPath $partial).Hash -ne $group.Hash) {
            throw "Checksum mismatch: $partial"
        }
        Move-Item -LiteralPath $partial -Destination $target
    }
}
Write-Host "STEP150_MODELS_STAGED $destination"
