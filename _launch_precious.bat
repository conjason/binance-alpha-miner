@echo off
cd /d D:\cl\MT5_AlphaGPT
"C:\Program Files\Python313\python.exe" train_precious_metals.py --offline > precious_metals_train.log 2>&1
