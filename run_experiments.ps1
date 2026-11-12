# Plano da segunda etapa: Self-Consistency, análise por rodadas, outro modelo e revisor Claude.
# Cada execução é retomável: se cair, rode o script de novo e ele continua de onde parou.
#
# Pré-requisitos:
#   arquivo .env nesta pasta com OPENAI_API_KEY=... (e ANTHROPIC_API_KEY=... para o revisor)
#   saídas antigas do gpt-4o-mini em .\legacy\ (o CoT de 1.000 tokens dentro de legacy\cot1000\)
#
# Uso:
#   .\run_experiments.ps1                       # etapas 1 a 3 (só OpenAI)
#   .\run_experiments.ps1 -WithReviewer         # + piloto do revisor Claude em 50 questões
#   .\run_experiments.ps1 -WithReviewer -ReviewerQuestions -1   # revisor no MedQA inteiro
param(
  [switch]$SkipLegacy,  # a pasta legacy\ vazia não faz nada; mantido para reaproveitar saídas no futuro
  [switch]$WithReviewer,
  [string]$ReviewerModel = 'claude-opus-5-5',
  [ValidateSet('low','medium','high','xhigh','max')][string]$ReviewerEffort = 'high',
  [int]$ReviewerQuestions = 50
)
$ErrorActionPreference = 'Continue'  # ver comentário em inference.ps1
Set-Location $PSScriptRoot
$py = '.\.venv\Scripts\python.exe'

# 1) gpt-4o-mini: importa o que já foi feito e preenche a resposta por rodada do MedAgents
if (-not $SkipLegacy) {
  $map = @{ 'syn_verif' = @('syn_verif', ''); 'base_direct' = @('base_direct', '');
            'base_cot' = @('base_cot', ''); 'cot1000' = @('base_cot', 'cot1000') }
  foreach ($f in Get-ChildItem .\legacy -File -Recurse) {
    # o CoT de 1.000 tokens foi gravado na pasta cot1000\ com o mesmo nome de arquivo do CoT de 300,
    # então o caminho completo é testado primeiro
    $hit = $null
    if ($f.FullName -like '*cot1000*') { $hit = 'cot1000' }
    else { $hit = @('syn_verif', 'base_direct', 'base_cot') | Where-Object { $f.Name -like "*$_*" } | Select-Object -First 1 }
    if (-not $hit) { Write-Warning "Ignorado (método não reconhecido): $($f.Name)"; continue }
    $method, $tag = $map[$hit]
    $argv = @('import_legacy.py', '--input', $f.FullName, '--model_name', 'gpt-4o-mini', '--method', $method)
    if ($tag) { $argv += @('--tag', $tag) }
    if ($method -eq 'syn_verif') { $argv += '--probe_rounds' }
    & $py @argv
    if ($LASTEXITCODE -ne 0) { throw "Falha ao importar $($f.Name)" }
  }
}

# 2 e 3) Os 5 métodos nas mesmas 1.273 questões, para cada modelo.
#    As saídas antigas do gpt-4o-mini não existem mais, então ele também roda tudo de novo.
#    Self-Consistency: 21 amostras (≈ chamadas do MedAgents), temperatura 0,7, 1.000 tokens.
foreach ($m in @('gpt-4o-mini', 'gpt-5.4-nano')) {
  .\inference.ps1 -Model $m -Method base_direct
  .\inference.ps1 -Model $m -Method base_cot
  .\inference.ps1 -Model $m -Method base_cot -CotMaxTokens 1000 -Tag cot1000
  .\inference.ps1 -Model $m -Method base_sc -CotMaxTokens 1000 -ScSamples 21
  .\inference.ps1 -Model $m -Method syn_verif
}

# 4) Opcional: MedAgents (gpt-4o-mini) + Claude como revisor externo.
#    Começa pelas primeiras 50 questões para medir o custo real (analyze.py mostra US$/questão).
if ($WithReviewer) {
  .\inference.ps1 -Model gpt-4o-mini -Method syn_verif_review -ReviewerModel $ReviewerModel `
      -ReviewerEffort $ReviewerEffort -EndPos $ReviewerQuestions
}

# 5) Tabelas e analysis\results.json
& $py analyze.py
