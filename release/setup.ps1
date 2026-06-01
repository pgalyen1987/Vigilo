# Vigilo Setup — Windows (PowerShell)
$ErrorActionPreference = "Stop"

$Version = "0.1.0"
$Image   = "vigilo-${Version}.tar.gz"

Write-Host "=============================="
Write-Host "  Vigilo ${Version} — Setup"
Write-Host "=============================="
Write-Host ""

# Check Docker
try {
    docker version | Out-Null
} catch {
    Write-Host "ERROR: Docker is not installed or not running." -ForegroundColor Red
    Write-Host "Install Docker Desktop from https://www.docker.com/products/docker-desktop"
    exit 1
}

Write-Host "[1/4] Creating directories..."
New-Item -ItemType Directory -Force -Path data, checkpoints/vigilo, reports | Out-Null

Write-Host "[2/4] Loading Docker image..."
if (Test-Path $Image) {
    docker load -i $Image
} else {
    Write-Host "WARNING: ${Image} not found. Place it here and re-run." -ForegroundColor Yellow
}

Write-Host "[3/4] Setting up configuration..."
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "  Created .env from template — edit it to point at your conn.log"
} else {
    Write-Host "  .env already exists, skipping"
}

Write-Host "[4/4] Done!"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Place your conn.log in the data\ folder"
Write-Host "  2. Edit .env and set VIGILO_LOG=data/your-file.conn.log"
Write-Host "  3. Run: docker compose up -d"
Write-Host "  4. Open: http://localhost:8088"
Write-Host ""
