@echo off
cd /d D:\cl\MT5_AlphaGPT
"C:\Program Files\Python313\python.exe" train_single.py US30.cash --offline > us30_train.log 2>&1
