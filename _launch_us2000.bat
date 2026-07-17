@echo off
cd /d D:\cl\MT5_AlphaGPT
"C:\Program Files\Python313\python.exe" train_single.py US2000.cash --offline > us2000_train.log 2>&1
