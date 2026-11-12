# Uso:
#   .\inference.ps1 -Method syn_verif -Model gpt-4o-mini
#   .\inference.ps1 -Method base_sc   -Model gpt-4o-mini -ScSamples 21
#   .\inference.ps1 -Method base_cot  -Model qwen2.5-7b-local -CotMaxTokens 1000 -Tag cot1000
#   .\inference.ps1 -Method syn_verif_review -Model gpt-4o-mini -ReviewerModel claude-haiku-5-5 -EndPos 50
# As chaves vêm do arquivo .env desta pasta (ou de variáveis de ambiente).
param(
  [ValidateSet('syn_verif','syn_verif_review','syn_only','anal_only','base_direct','base_cot','base_sc')]
  [string]$Method = 'syn_verif',
  [string]$Model = 'gpt-4o-mini',
  [int]$CotMaxTokens = 300,
  [int]$ScSamples = 21,
  [double]$ScTemperature = 0.7,
  [int]$Workers = 8,
  [int]$StartPos = 0,
  [int]$EndPos = -1,
  [string]$Tag = '',
  [string]$ReviewerModel = 'claude-opus-5-5',
  [ValidateSet('low','medium','high','xhigh','max')][string]$ReviewerEffort = 'high',
  [string]$OutputFolder = ''
)
# 'Continue': no PowerShell 5.1 a barra de progresso (stderr) seria tratada como erro fatal
$ErrorActionPreference = 'Continue'
Set-Location $PSScriptRoot
if (-not $OutputFolder) { $OutputFolder = "./outputs/MedQA/$($Model -replace '[/:]','_')/" }
$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$argv = @('-u','run.py','--model_name',$Model,'--method',$Method,'--dataset_name','MedQA',
          '--dataset_dir','./datasets/MedQA/','--start_pos',$StartPos,'--end_pos',$EndPos,
          '--output_files_folder',$OutputFolder,'--max_attempt_vote',3,
          '--cot_max_tokens',$CotMaxTokens,'--sc_samples',$ScSamples,'--sc_temperature',$ScTemperature,
          '--workers',$Workers)
if ($Tag) { $argv += @('--tag',$Tag) }
if ($Method -eq 'syn_verif_review') { $argv += @('--reviewer_model',$ReviewerModel,'--reviewer_effort',$ReviewerEffort) }
& $py @argv
if ($LASTEXITCODE -ne 0) { throw "run.py terminou com erro ($LASTEXITCODE): $Model $Method" }
