# Dedicated hidden Word instance; never attaches to or closes the user's documents.
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$documentRoot = Join-Path $taskRoot 'ITAI2373-NewsBot-Midterm\documents'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($source in Get-ChildItem -LiteralPath $documentRoot -Filter '*.docx') {
        $doc = $word.Documents.Open($source.FullName, $false, $true)
        try {
            $doc.Repaginate()
            $pages = $doc.ComputeStatistics(2)
            $pdfPath = [System.IO.Path]::ChangeExtension($source.FullName, '.pdf')
            $doc.ExportAsFixedFormat($pdfPath, 17)
            Write-Output "$($source.Name): $pages pages"
        } finally { $doc.Close(0) }
    }
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
