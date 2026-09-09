Set-Location -Path $PSScriptRoot
& ".\venv\Scripts\streamlit.exe" run app.py --server.headless true
