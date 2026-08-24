[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$repoRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$submodulePath = '.agents/sources/agent-skills'
$submoduleRoot = Join-Path $repoRoot $submodulePath
$targetRoot = Join-Path $repoRoot '.agents/skills'

$git = (Get-Command git -ErrorAction Stop).Source
& $git -C $repoRoot submodule sync -- $submodulePath
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to synchronize the agent-skills submodule configuration.'
}

& $git -C $repoRoot submodule update --init -- $submodulePath
if ($LASTEXITCODE -ne 0) {
    throw 'Failed to initialize the agent-skills submodule.'
}

$linkScript = Join-Path $submoduleRoot 'scripts/link-skills.ps1'
if (-not (Test-Path -LiteralPath $linkScript)) {
    throw "Missing linker in initialized submodule: $linkScript"
}

& $linkScript -TargetDirectory $targetRoot
