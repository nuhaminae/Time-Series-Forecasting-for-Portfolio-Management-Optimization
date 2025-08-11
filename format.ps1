# Windows PowerShell version (format.ps1)
# Run .\format.ps1 in bash
.\.tslvenv\Scripts\black . --exclude .tslvenv
.\.tslvenv\Scripts\isort . --skip .tslvenv
.\.tslvenv\Scripts\python.exe -m flake8 . --exclude=.tslvenv
if (Get-ChildItem -Recurse -Filter *.ipynb) {
  Write-Host "Formatting notebooks with black and isort via nbqa..."
  .\.tslvenv\Scripts\python.exe -m nbqa black . --line-length=88 --exclude .tslvenv
  .\.tslvenv\Scripts\python.exe -m nbqa isort . --skip .tslvenv
  Write-Host "Linting notebooks with flake8 via nbqa..."
  .\.tslvenv\Scripts\python.exe -m nbqa flake8 . --exclude .tslvenv --exit-zero
} else {
  Write-Host "No notebooks found - skipping nbQA."
}
