@echo off
setlocal enabledelayedexpansion

for %%i in (1 2 3 4 5 6 7) do (
    mkdir "LaMP_%%i"
    cd "LaMP_%%i"
    for %%j in (train dev) do (
        curl -O "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_%%i/%%j/%%j_questions.json"
        curl -O "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_%%i/%%j/%%j_outputs.json"
    )
    curl -O "https://ciir.cs.umass.edu/downloads/LaMP/LaMP_%%i/test/test_questions.json"
    cd ..
)

endlocal
