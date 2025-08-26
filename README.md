# 🔎 Local Search

A lightweight, privacy-first command-line tool to **index and search your local files** (text, markdown, CSV, and more).  
Everything runs **100% offline** — no cloud services, no data sharing.

---

## ✨ Features
- 📂 **File Indexing**: Quickly scan a directory and build a searchable index.  
- 🔍 **Fast Search**: Find keywords and phrases instantly across multiple files.  
- 🔒 **Privacy-First**: All searches stay local; nothing leaves your computer.  
- 📑 **Cross-Format Support**: Works with `.txt`, `.md`, `.csv`, and `.pdf` (optional).  
- 🖥️ **Simple CLI**: Use straightforward commands from the terminal.  

---

## 🛠️ Tech Stack
- **Python 3** for core logic + CLI interface  
- **Regex & Text Parsing** for keyword/phrase matching  
- **CSV module** for spreadsheets  
- **PyPDF2** (optional) for PDF text extraction  

---

## 🚀 Usage

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/local-search.git
cd local-search
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Index your files
```powershell
python local_search.py index C:\Users\Andrew\Documents
```

### 4️⃣ Search your files
```powershell
python local_search.py search "machine learning"
```

---

## 📸 Screenshots
*(Add your own screenshots here of indexing and searching results in the terminal)*  

---

## 🎯 Why This Project?
- Demonstrates **systems design**: indexing + searching logic  
- Practical and **portfolio-ready** — everyone needs to search files  
- Highlights **Python fundamentals**: file I/O, regex, parsing, error handling  
- A unique alternative to overused “to-do list” projects  

---

## 📄 License
MIT License – feel free to use and adapt.  
