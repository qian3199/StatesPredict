# 整理Dytr_AIdone项目文件
$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $baseDir

Write-Host "开始整理文件... (当前目录: $baseDir)" -ForegroundColor Green

# 1. 将根目录下的pkl文件移动到data目录
Write-Host "`n1. 移动根目录下的pkl文件到data目录..." -ForegroundColor Yellow
$pklFiles = Get-ChildItem -Path ".\*.pkl" -File -ErrorAction SilentlyContinue
if ($pklFiles) {
    foreach ($file in $pklFiles) {
        $destPath = Join-Path ".\data" $file.Name
        Write-Host "移动 $($file.Name) -> data\"
        Move-Item -Path $file.FullName -Destination $destPath -Force
    }
} else {
    Write-Host "未找到pkl文件"
}

# 2. 整理outputs目录下的训练图片
Write-Host "`n2. 整理outputs目录下的训练图片..." -ForegroundColor Yellow
if (Test-Path ".\outputs") {
    $outputDirs = Get-ChildItem -Path ".\outputs" -Directory | Sort-Object Name

    foreach ($dir in $outputDirs) {
        Write-Host "`n处理目录: $($dir.Name)"
        
        # 创建对应的训练结果目录
        $trainDir = Join-Path ".\images" $dir.Name
        if (-not (Test-Path $trainDir)) {
            New-Item -ItemType Directory -Path $trainDir -Force | Out-Null
            Write-Host "创建目录: $($dir.Name)"
        }
        
        # 复制所有图片到训练目录
        $imageFiles = Get-ChildItem -Path $dir.FullName -File -Include "*.png", "*.jpg", "*.jpeg" -Recurse -ErrorAction SilentlyContinue
        foreach ($img in $imageFiles) {
            $destImg = Join-Path $trainDir $img.Name
            Copy-Item -Path $img.FullName -Destination $destImg -Force
        }
        Write-Host "复制了 $($imageFiles.Count) 个图片文件"
    }
} else {
    Write-Host "outputs目录不存在"
}

Write-Host "`n整理完成!" -ForegroundColor Green
