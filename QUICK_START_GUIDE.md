# Quick Start Guide - Run Your Model Tomorrow

## 🚀 Simple Steps to Get Started

### Step 1: Open PowerShell
- Press `Windows Key + X`
- Click "Windows PowerShell" or "Terminal"

### Step 2: Navigate to Your Project
```powershell
cd C:\Agritech_ML
```

### Step 3: Activate Virtual Environment
```powershell
.\venv\Scripts\activate.ps1
```

You should see `(venv)` appear at the start of your prompt.

---

## Option A: Quick Test (No Server Needed) ✅ RECOMMENDED

### Test Your Model:
```powershell
python test_predict.py --format pretty
```

### Test with Location (Random Dataset Selection):
```powershell
python test_predict.py --realtime --lat 28.6 --lon 77.2 --format pretty
```

### Or Use Batch File (Easiest):
```powershell
.\test_all_local.bat
```
Just double-click `test_all_local.bat` in File Explorer!

---

## Option B: Start Server and Test API

### Step 1: Start the Server
```powershell
python main.py
```

**Keep this terminal open!** You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Step 2: Open a NEW Terminal Window
- Press `Windows Key + X` again
- Open another PowerShell window

### Step 3: Test the API
```powershell
cd C:\Agritech_ML
.\venv\Scripts\activate.ps1
python test_api_endpoints.py
```

---

## 📋 Complete Command List (Copy & Paste)

### For Quick Testing (No Server):
```powershell
cd C:\Agritech_ML
.\venv\Scripts\activate.ps1
python test_predict.py --format pretty
```

### For API Testing (With Server):
```powershell
# Terminal 1:
cd C:\Agritech_ML
.\venv\Scripts\activate.ps1
python main.py

# Terminal 2 (new window):
cd C:\Agritech_ML
.\venv\Scripts\activate.ps1
python test_api_endpoints.py
```

---

## 🎯 What Each Command Does

| Command | What It Does |
|---------|-------------|
| `cd C:\Agritech_ML` | Go to your project folder |
| `.\venv\Scripts\activate.ps1` | Activate Python environment |
| `python test_predict.py` | Test predictions (local, no server) |
| `python main.py` | Start the API server |
| `.\test_all_local.bat` | Run all tests automatically |

---

## ⚠️ Common Issues & Solutions

### Issue: "Python was not found"
**Solution:** Make sure you activated the virtual environment:
```powershell
.\venv\Scripts\activate.ps1
```

### Issue: "Connection refused" when using `--http`
**Solution:** Either:
1. Start the server first: `python main.py`
2. Or remove `--http` flag (use local mode)

### Issue: "Module not found"
**Solution:** Make sure virtual environment is activated and dependencies are installed:
```powershell
.\venv\Scripts\activate.ps1
pip install -r requirements.txt
```

---

## 🎯 Recommended Workflow for Tomorrow

### Morning Routine:
1. Open PowerShell
2. Run: `cd C:\Agritech_ML`
3. Run: `.\venv\Scripts\activate.ps1`
4. Run: `.\test_all_local.bat` (or double-click it)

**That's it!** Your model is tested and ready.

---

## 📝 Quick Reference Card

**Copy this and save it:**

```
QUICK START:
1. cd C:\Agritech_ML
2. .\venv\Scripts\activate.ps1
3. python test_predict.py --format pretty

OR JUST:
Double-click test_all_local.bat
```

---

## ✅ Checklist for Tomorrow

- [ ] Open PowerShell
- [ ] Navigate to project: `cd C:\Agritech_ML`
- [ ] Activate environment: `.\venv\Scripts\activate.ps1`
- [ ] Run test: `python test_predict.py --format pretty`
- [ ] See predictions! ✅

---

**That's all you need!** Your model is ready to use. 🎉


