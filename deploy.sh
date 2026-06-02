#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — QuantEdge AI  ·  GitHub + Streamlit Cloud deployment helper
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh
#
# What this script does:
#   1. Validates you're in the right directory
#   2. Checks git is installed
#   3. Initialises a git repo (if not already one)
#   4. Configures .gitignore safety check
#   5. Stages all files and makes the initial commit
#   6. Prompts for your GitHub repo URL and pushes
#   7. Prints the exact Streamlit Cloud deployment instructions
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
BLUE='\033[0;34m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
RESET='\033[0m'

log()  { echo -e "${CYAN}▶${RESET}  $*"; }
ok()   { echo -e "${GREEN}✓${RESET}  $*"; }
warn() { echo -e "${YELLOW}⚠${RESET}  $*"; }
err()  { echo -e "${RED}✗${RESET}  $*" >&2; exit 1; }
hr()   { echo -e "${BLUE}────────────────────────────────────────────────────────────${RESET}"; }

hr
echo -e "${BOLD}${CYAN}  ⚡  QuantEdge AI — Deployment Script${RESET}"
echo -e "  Streamlit Cloud · GitHub · Cougar Global Methodology"
hr
echo ""

# ── Step 0: Sanity checks ────────────────────────────────────────────────────
log "Checking environment..."

if ! command -v git &>/dev/null; then
    err "git is not installed. Install it from https://git-scm.com and re-run."
fi

REQUIRED_FILES=("app.py" "portfolio_engine.py" "analyst.py" "requirements.txt" ".streamlit/config.toml")
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        err "Required file missing: $f — make sure you're running this from the quantedge-ai folder."
    fi
done
ok "All required files present"

# Safety: ensure .env is gitignored
if [[ -f ".env" ]]; then
    if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
        warn ".env exists but is NOT in .gitignore — adding it now."
        echo ".env" >> .gitignore
    fi
    ok ".env will NOT be committed (gitignored)"
fi

# ── Step 1: Git init ─────────────────────────────────────────────────────────
echo ""
log "Initialising git repository..."

if [[ -d ".git" ]]; then
    warn "Git repo already exists — skipping git init."
else
    git init
    ok "Git repository initialised"
fi

# Set default branch to main
git checkout -b main 2>/dev/null || git checkout main 2>/dev/null || true
ok "Branch: main"

# ── Step 2: Stage & commit ───────────────────────────────────────────────────
echo ""
log "Staging files..."

# Stage everything (respects .gitignore automatically)
git add .

# Show what's staged
echo ""
echo -e "${BOLD}Files to be committed:${RESET}"
git diff --cached --name-only | while read -r line; do
    echo -e "  ${GREEN}+${RESET} $line"
done
echo ""

log "Creating initial commit..."
git commit -m "🚀 Initial commit: QuantEdge AI portfolio analyst

- Streamlit dark-theme app (Bloomberg aesthetic)
- Cougar Global tactical allocation methodology
- Post Modern Portfolio Theory (PMPT) risk metrics
- Probability of Loss as primary risk metric
- 20-asset-class macro framework (Bull/Base/Bear)
- Monte Carlo simulation with downside risk bands
- Claude AI analyst integration (graceful offline fallback)
- Live market data via yfinance" \
    --allow-empty 2>/dev/null || true

ok "Initial commit created"

# ── Step 3: Add GitHub remote ────────────────────────────────────────────────
echo ""
hr
echo -e "${BOLD}  Step 3: Connect to GitHub${RESET}"
hr
echo ""
echo -e "  Create a ${BOLD}new empty repo${RESET} on GitHub first:"
echo -e "  ${CYAN}→ https://github.com/new${RESET}"
echo -e "  ${YELLOW}• Repository name:${RESET}  quantedge-ai"
echo -e "  ${YELLOW}• Visibility:${RESET}        Public  (required for free Streamlit Cloud)"
echo -e "  ${YELLOW}• Initialise:${RESET}        ✗ None  (don't add README/gitignore — we have ours)"
echo ""

read -r -p "  Paste your GitHub repo URL (e.g. https://github.com/YourName/quantedge-ai): " REPO_URL

if [[ -z "$REPO_URL" ]]; then
    warn "No URL entered. Skipping remote setup."
    warn "Run manually later:  git remote add origin <URL> && git push -u origin main"
else
    # Remove existing origin if present
    git remote remove origin 2>/dev/null || true
    git remote add origin "$REPO_URL"
    ok "Remote 'origin' set to: $REPO_URL"

    echo ""
    log "Pushing to GitHub..."
    if git push -u origin main; then
        ok "Pushed to GitHub successfully"
    else
        warn "Push failed. You may need to authenticate. Try:"
        echo "    git push -u origin main"
        echo "  If using HTTPS, GitHub now requires a Personal Access Token instead of password."
        echo "  Generate one at: https://github.com/settings/tokens"
    fi
fi

# ── Step 4: Streamlit Cloud instructions ─────────────────────────────────────
echo ""
hr
echo -e "${BOLD}  Step 4: Deploy on Streamlit Cloud${RESET}"
hr
echo ""
echo -e "  ${BOLD}1. Open:${RESET}  ${CYAN}https://share.streamlit.io${RESET}"
echo -e "  ${BOLD}2. Click:${RESET} \"New app\""
echo -e "  ${BOLD}3. Fill in:${RESET}"
echo -e "     ${YELLOW}Repository:${RESET}   your GitHub repo  (e.g. YourName/quantedge-ai)"
echo -e "     ${YELLOW}Branch:${RESET}       main"
echo -e "     ${YELLOW}Main file:${RESET}    app.py"
echo ""
echo -e "  ${BOLD}4. Before clicking Deploy — add your API key:${RESET}"
echo -e "     Click ${YELLOW}\"Advanced settings\"${RESET} → ${YELLOW}\"Secrets\"${RESET}"
echo -e "     Paste exactly:"
echo ""
echo -e "     ${CYAN}ANTHROPIC_API_KEY = \"sk-ant-your-key-here\"${RESET}"
echo ""
echo -e "     (Replace with your real key from ${CYAN}https://console.anthropic.com${RESET})"
echo ""
echo -e "  ${BOLD}5. Click Deploy.${RESET}"
echo -e "     Streamlit Cloud installs requirements.txt automatically."
echo -e "     First deploy takes ~2 minutes. Your app URL will be:"
echo -e "     ${CYAN}https://YourName-quantedge-ai-app-XXXX.streamlit.app${RESET}"
echo ""
echo -e "  ${BOLD}6. Update your README.md badge:${RESET}"
echo -e "     Replace the placeholder badge URL with your real app URL."
echo ""
hr
echo -e "${GREEN}${BOLD}  ✓  QuantEdge AI is ready to deploy!${RESET}"
hr
echo ""
