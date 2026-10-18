<#
.SYNOPSIS
    把上游 sorryhyun/anima_lora 合进本仓库（下游整合版）的本地脚本。

.DESCRIPTION
    和 .github/workflows/upstream-sync.yml 是同一件事的两份实现：CI 每天自动开 PR，
    本脚本给本机用（想立刻看 drift / 想手动合 / 想推回 origin）。

    为什么要有这个脚本：本仓库是「上游为基线 + 极小的自有面」，
    维护成本全押在"合并是否还干净"上。所以这里只做三件事：
      1. 报 drift（落后上游几个提交、上游改了什么）；
      2. 合并（冲突就停下并把冲突文件点名，让人只在那几个文件上动手）；
      3. 跑守卫测试（我们依赖的上游缝合点还在不在、补丁面有没有偷偷变大）。

    用法：
        pwsh -File scripts\downstream\upstream-sync.ps1              # 只读：报 drift
        pwsh -File scripts\downstream\upstream-sync.ps1 -Merge       # 合并 + 跑守卫
        pwsh -File scripts\downstream\upstream-sync.ps1 -Merge -Push # 再推回 origin
        pwsh -File scripts\downstream\upstream-sync.ps1 -Py <python.exe>   # 指定跑守卫的解释器

    出网：本机公网走 127.0.0.1:7890（见仓库 AGENTS），脚本自己带上
    HTTPS_PROXY/HTTP_PROXY（已有则不覆盖），否则 git fetch 在部分网络下会挂。
#>
param(
    [switch]$Merge,
    [switch]$Push,
    [string]$Python = "",
    [string]$UpstreamUrl = "https://github.com/sorryhyun/anima_lora.git"
)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $Root

if (-not $env:HTTPS_PROXY) { $env:HTTPS_PROXY = 'http://127.0.0.1:7890' }
if (-not $env:HTTP_PROXY) { $env:HTTP_PROXY = 'http://127.0.0.1:7890' }

function Say($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Warn($m) { Write-Host "warning: $m" -ForegroundColor Yellow }

# 1. upstream remote（缺了就补上，不覆盖已有的）
$remotes = git remote
if ($remotes -notcontains 'upstream') {
    Say "添加 upstream remote -> $UpstreamUrl"
    git remote add upstream $UpstreamUrl
}

Say 'git fetch upstream main'
git fetch --no-tags upstream main

# 2. drift
$counts = (git rev-list --left-right --count HEAD...upstream/main) -split '\s+'
$ahead = [int]$counts[0]; $behind = [int]$counts[1]
$base = if (Test-Path '.downstream/upstream-base') { (Get-Content '.downstream/upstream-base' -Raw).Trim() } else { '' }
$upstreamHead = (git rev-parse upstream/main).Trim()
Write-Host ""
Write-Host ("本地领先 {0} 个提交，落后上游 {1} 个提交" -f $ahead, $behind)
Write-Host ("基线记录 .downstream/upstream-base = {0}" -f ($(if ($base) { $base.Substring(0,12) } else { '(无)' })))
Write-Host ("上游当前 main                    = {0}" -f $upstreamHead.Substring(0,12))
if ($behind -eq 0) { Say '已经跟上游一致，什么都不用做'; return 0 }

if (-not $Merge) {
    Write-Host ""
    Say '上游这期间的提交（最多 40 条）：'
    git log --oneline --no-decorate "HEAD..upstream/main" | Select-Object -First 40
    Write-Host ""
    Warn '这是只读模式；要合并请加 -Merge'
    return 0
}

# 3. merge
Say 'git merge upstream/main'
git merge --no-edit upstream/main
if ($LASTEXITCODE -ne 0) {
    $conflicts = (git diff --name-only --diff-filter=U) -join ', '
    Write-Host ""
    Write-Host "合并有冲突：$conflicts" -ForegroundColor Red
    $patched = Get-Content '.downstream/patched-upstream-files.txt' |
        Where-Object { $_ -and -not $_.StartsWith('#') }
    $hit = ($conflicts -split ',\s*') | Where-Object { $patched -contains $_ }
    if ($hit) { Write-Host ("其中属于我们的补丁面：{0}" -f ($hit -join ', ')) -ForegroundColor Yellow }
    Write-Host '解决冲突后：git add <files>; git commit；然后重跑本脚本（-Merge）或跑守卫测试。'
    return 1
}
Say '合并干净'

# 4. 基线前进 + 守卫
Set-Content -LiteralPath '.downstream/upstream-base' -Value $upstreamHead -Encoding ascii
Say "更新 .downstream/upstream-base -> $($upstreamHead.Substring(0,12))"

if (-not $Python) {
    $cand = @('.venv\Scripts\python.exe', 'E:\anima_lora\.venv\Scripts\python.exe')
    $Python = ($cand | Where-Object { Test-Path $_ } | Select-Object -First 1)
}
if ($Python) {
    Say "跑守卫测试（$Python）"
    & $Python -m pytest tests/test_downstream_guards.py tests/test_gui_i18n_parity.py -q
    if ($LASTEXITCODE -ne 0) { Warn '守卫测试没过 —— 先修，别推'; return 1 }
} else {
    Warn '没找到 python 解释器，跳过守卫测试（-Py <python.exe> 可指定）'
}

# 5. 提交基线前进（如果只有基线文件变了）
git add .downstream/upstream-base
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -m "chore(upstream): advance baseline to $($upstreamHead.Substring(0,12))"
}

if ($Push) {
    Say 'git push origin HEAD'
    git push origin HEAD
}
Say '完成'
return 0
