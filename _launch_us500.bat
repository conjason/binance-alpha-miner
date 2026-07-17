@echo off
cd /d D:\cl\MT5_AlphaGPT
"C:\Program Files\Python313\python.exe" train_single.py US500.cash --offline > us500_train.log 2>&1
