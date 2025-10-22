# DeFi Risk Monitor - Complete Startup Script
# Save as: startup.ps1
# Run with: .\startup.ps1

Write-Host "================================" -ForegroundColor Cyan
Write-Host "DeFi Risk Monitor - Startup" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check PostgreSQL
Write-Host "Step 1: Checking PostgreSQL..." -ForegroundColor Yellow
$pgService = Get-Service -Name "postgresql*" -ErrorAction SilentlyContinue

if ($pgService -and $pgService.Status -eq "Running") {
    Write-Host "✅ PostgreSQL is running" -ForegroundColor Green
} else {
    Write-Host "❌ PostgreSQL is not running" -ForegroundColor Red
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Yellow
    Write-Host "1. Start PostgreSQL service (if installed):" -ForegroundColor White
    Write-Host "   Start-Service postgresql-x64-15" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Or use Railway PostgreSQL:" -ForegroundColor White
    Write-Host "   Set DATABASE_URL in .env file" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Or continue with SQLite (remove DATABASE_URL from .env)" -ForegroundColor White
    Write-Host ""
    
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit
    }
}

# Step 2: Check virtual environment
Write-Host ""
Write-Host "Step 2: Checking virtual environment..." -ForegroundColor Yellow

if (Test-Path ".\DefiLiquidation\Scripts\Activate.ps1") {
    Write-Host "✅ Virtual environment found" -ForegroundColor Green
    & ".\DefiLiquidation\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} else {
    Write-Host "❌ Virtual environment not found" -ForegroundColor Red
    Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
    python -m venv DefiLiquidation
    & ".\DefiLiquidation\Scripts\Activate.ps1"
    Write-Host "✅ Virtual environment created and activated" -ForegroundColor Green
}

# Step 3: Install/Update dependencies
Write-Host ""
Write-Host "Step 3: Checking dependencies..." -ForegroundColor Yellow
Write-Host "Installing/Updating packages..." -ForegroundColor Gray
pip install -q -r requirements.txt

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Dependencies installed" -ForegroundColor Green
} else {
    Write-Host "⚠️ Some dependencies failed to install" -ForegroundColor Yellow
}

# Step 4: Check .env file
Write-Host ""
Write-Host "Step 4: Checking environment variables..." -ForegroundColor Yellow

if (Test-Path ".env") {
    Write-Host "✅ .env file found" -ForegroundColor Green
    
    # Check for required variables
    $envContent = Get-Content .env -Raw
    
    $requiredVars = @(
        "DUNE_API_KEY_CURRENT_POSITION",
        "DUNE_API_KEY_LIQUIDATION_HISTORY"
    )
    
    $missingVars = @()
    foreach ($var in $requiredVars) {
        if ($envContent -notmatch $var) {
            $missingVars += $var
        }
    }
    
    if ($missingVars.Count -gt 0) {
        Write-Host "⚠️ Missing environment variables:" -ForegroundColor Yellow
        foreach ($var in $missingVars) {
            Write-Host "   - $var" -ForegroundColor Red
        }
    } else {
        Write-Host "✅ All required variables present" -ForegroundColor Green
    }
} else {
    Write-Host "❌ .env file not found" -ForegroundColor Red
    Write-Host "Please create .env file with required variables" -ForegroundColor Yellow
    exit
}

# Step 5: Initialize database
Write-Host ""
Write-Host "Step 5: Initializing database..." -ForegroundColor Yellow

try {
    python -c "from app.db_models import Base, engine; Base.metadata.create_all(engine); print('✅ Database tables created')"
    Write-Host "✅ Database initialized" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Database initialization failed (may already be initialized)" -ForegroundColor Yellow
}

# Step 6: Check data
Write-Host ""
Write-Host "Step 6: Checking existing data..." -ForegroundColor Yellow

try {
    $dataCheck = python -c @"
from app.db_models import SessionLocal, Reserve, Position
db = SessionLocal()
reserves = db.query(Reserve).count()
positions = db.query(Position).count()
db.close()
print(f'{reserves},{positions}')
"@

    $counts = $dataCheck -split ','
    $reserves = [int]$counts[0]
    $positions = [int]$counts[1]
    
    Write-Host "   Reserves: $reserves" -ForegroundColor Gray
    Write-Host "   Positions: $positions" -ForegroundColor Gray
    
    if ($reserves -eq 0 -or $positions -eq 0) {
        Write-Host "⚠️ Database is empty" -ForegroundColor Yellow
        Write-Host "After starting, run data refresh:" -ForegroundColor Gray
        Write-Host "   POST http://localhost:8080/api/data/refresh" -ForegroundColor Gray
    } else {
        Write-Host "✅ Data found in database" -ForegroundColor Green
    }
} catch {
    Write-Host "⚠️ Could not check data (database may not be accessible)" -ForegroundColor Yellow
}

# Step 7: Start the application
Write-Host ""
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Starting Application..." -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Server will start at: http://localhost:8080" -ForegroundColor Green
Write-Host "API Documentation: http://localhost:8080/docs" -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start uvicorn
uvicorn app.main:app --reload --port 8080 --host 0.0.0.0