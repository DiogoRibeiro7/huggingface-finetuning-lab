param(
    [Parameter(Mandatory = $true)]
    [string]$Repo,
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI ('gh') is required. Install from https://cli.github.com/"
}

$body = @{
    required_status_checks = @{
        strict = $true
        contexts = @(
            "quality (3.10)",
            "quality (3.11)"
        )
    }
    enforce_admins = $true
    required_pull_request_reviews = @{
        required_approving_review_count = 1
        dismiss_stale_reviews = $true
        require_code_owner_reviews = $true
        require_last_push_approval = $false
    }
    restrictions = $null
    required_linear_history = $false
    allow_force_pushes = $false
    allow_deletions = $false
    block_creations = $false
    required_conversation_resolution = $true
    lock_branch = $false
    allow_fork_syncing = $true
} | ConvertTo-Json -Depth 8

Write-Host "Applying branch protection to $Repo:$Branch ..."
$body | gh api `
    --method PUT `
    -H "Accept: application/vnd.github+json" `
    "/repos/$Repo/branches/$Branch/protection" `
    --input - | Out-Null

Write-Host "Branch protection updated successfully."
