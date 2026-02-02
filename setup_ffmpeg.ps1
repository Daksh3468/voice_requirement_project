Write-Host "Downloading FFmpeg..."
$url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
$output = "ffmpeg.zip"

Invoke-WebRequest -Uri $url -OutFile $output

Write-Host "Extracting FFmpeg..."
Expand-Archive -Path $output -DestinationPath "ffmpeg_temp" -Force

Write-Host "Setting up..."
$binPath = Get-ChildItem -Path "ffmpeg_temp" -Recurse -Filter "ffmpeg.exe" | Select-Object -ExpandProperty DirectoryName
if (-not (Test-Path "bin")) {
    New-Item -ItemType Directory -Path "bin" | Out-Null
}

Copy-Item -Path "$binPath\ffmpeg.exe" -Destination "bin\ffmpeg.exe"
Copy-Item -Path "$binPath\ffprobe.exe" -Destination "bin\ffprobe.exe"

Write-Host "Cleanup..."
Remove-Item -Path "ffmpeg_temp" -Recurse -Force
Remove-Item -Path $output -Force

Write-Host "FFmpeg installed to .\bin\"
Write-Host "You can now run the app."
