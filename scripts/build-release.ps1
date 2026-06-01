# Vigilo release builder (PowerShell — runs on Windows, macOS, Linux)
param(
    [string]$Version = "0.1.0"
)

$ErrorActionPreference = "Stop"

$ImageName  = "vigilo"
$ImageTag   = "${ImageName}:${Version}"
$ReleaseDir = "release"

Write-Host "==> Building ${ImageTag}"
docker build -t $ImageTag -t "${ImageName}:latest" .
if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }

Write-Host "==> Exporting image to ${ReleaseDir}/"
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null
docker save $ImageTag | gzip > "${ReleaseDir}/${ImageName}-${Version}.tar.gz"

Write-Host "==> Packaging customer files"
Copy-Item -ErrorAction SilentlyContinue "release/docker-compose.yml" $ReleaseDir
Copy-Item -ErrorAction SilentlyContinue "release/README.md" $ReleaseDir
Copy-Item ".env.example" "${ReleaseDir}/.env.example"

$Size = (Get-Item "${ReleaseDir}/${ImageName}-${Version}.tar.gz").Length / 1MB
$SizeMB = [math]::Round($Size, 1)

Write-Host ""
Write-Host "==> Release built:"
Write-Host "    Image:   ${ReleaseDir}/${ImageName}-${Version}.tar.gz (${SizeMB} MB)"
Write-Host "    Compose: ${ReleaseDir}/docker-compose.yml"
Write-Host "    Docs:    ${ReleaseDir}/README.md"
Write-Host ""
Write-Host "Ship the release/ folder to customers."
Write-Host "They run: docker load -i vigilo-${Version}.tar.gz ; docker compose up -d"
