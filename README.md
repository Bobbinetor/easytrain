# 🎯 EasyTrain

<div align="center">

**Transform Documents into High-Quality LLM Training Data**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Powered%20by-Ollama-ff6b6b.svg)](https://ollama.com)

*Convert CSV, Excel, Word, PDF, and PowerPoint documents into knowledge-rich JSON datasets for LLM fine-tuning*

</div>

---

## 📋 Overview

EasyTrain is a Python toolkit that intelligently converts multi-format documents into structured JSON datasets optimized for LLM training. Instead of simply extracting document content, EasyTrain focuses on **extracting generalizable knowledge, principles, and expertise** that can enhance an LLM's capabilities across domains.

### 🚧 Development Status

**Note**: EasyTrain is currently under active development. While we acknowledge the existence of other document conversion tools, EasyTrain's core strength lies in its **simplicity and focus on research workflows**. Our goal is to provide researchers and practitioners with a straightforward, locally-controlled solution for creating high-quality training datasets from their documents.

### 👨‍💻 Author

**Alfredo Petruolo** - *Research & Development*

---

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🔄 **Universal Document Processing**
- **Multi-format Support**: CSV, Excel (.xlsx/.xls), Word (.docx/.doc), PDF, PowerPoint (.pptx/.ppt)
- **Intelligent Detection**: Automatic format recognition with MIME type validation
- **Structure Preservation**: Maintains document hierarchy and relationships

</td>
<td width="50%">

### 🧠 **Knowledge-Focused Extraction**
- **Expert Knowledge Mining**: Extracts principles, methodologies, and best practices
- **Domain Expertise**: Creates learning-oriented Q&A pairs
- **Transferable Insights**: Generates knowledge applicable beyond source documents

</td>
</tr>
<tr>
<td width="50%">

### ⚡ **Smart Processing**
- **Adaptive Chunking**: Context-aware document segmentation
- **Local LLM Integration**: Powered by Ollama for privacy and control
- **Robust Error Handling**: Graceful failure recovery with detailed logging

</td>
<td width="50%">

### 🎯 **Research-Ready Output**
- **Training-Optimized Format**: Instruction-input-output triplets
- **Quality Validation**: Schema compliance and data integrity checks
- **Flexible Export**: JSON and JSONL format support

</td>
</tr>
</table>

---

## 🛠️ System Requirements

### **Minimum Requirements**
- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB+ recommended for large documents)
- **Storage**: 2GB free space for models and processing

### **Required Dependencies for Full Functionality**
- **Java**: Required for PDF table extraction (tabula-py)
- **Poppler**: Required for PDF text extraction and OCR preprocessing
- **Tesseract OCR**: Required for optical character recognition in scanned PDFs/images

### **Platform Support**

<details>
<summary><strong>🐧 Linux (Ubuntu/Debian)</strong></summary>

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install python3 python3-pip python3-venv -y

# Install Java (required for PDF table extraction)
sudo apt install default-jre -y

# Install Poppler (required for PDF processing)
sudo apt install poppler-utils -y

# Install Tesseract OCR (required for scanned document processing)
sudo apt install tesseract-ocr tesseract-ocr-eng -y

# Install additional language packs if needed
# sudo apt install tesseract-ocr-ita tesseract-ocr-fra tesseract-ocr-deu

# Install additional dependencies
sudo apt install build-essential libffi-dev -y
```
</details>

<details>
<summary><strong>🍎 macOS</strong></summary>

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Install Java
brew install openjdk

# Install Poppler (required for PDF processing)
brew install poppler

# Install Tesseract OCR (required for scanned document processing)
brew install tesseract

# Install additional language packs if needed
# brew install tesseract-lang

# Install additional dependencies
brew install cairo pkg-config
```
</details>

<details>
<summary><strong>🪟 Windows</strong></summary>

1. **Install Python 3.8+** from [python.org](https://python.org)
2. **Install Java** from [Oracle](https://www.oracle.com/java/technologies/downloads/) or [OpenJDK](https://openjdk.org/)
3. **Install Poppler** for PDF processing:
   - Download from [poppler.freedesktop.org](https://poppler.freedesktop.org/)
   - Or use conda: `conda install -c conda-forge poppler`
   - Or use chocolatey: `choco install poppler`
4. **Install Tesseract OCR** for scanned document processing:
   - Download from [tesseract-ocr.github.io](https://tesseract-ocr.github.io/tessdoc/Downloads.html)
   - Or use chocolatey: `choco install tesseract`
5. **Add Java, Poppler, and Tesseract to PATH** environment variable
6. **Install Microsoft Visual C++ Build Tools** (for some dependencies)

</details>

---

## 🚀 Getting Started

### **Step 1: Install Ollama**

<details>
<summary>📱 <strong>Installation Instructions</strong></summary>

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
brew install ollama
```

**Windows:**
Download and install from [ollama.com](https://ollama.com)

</details>

### **Step 2: Pull an LLM Model**

```bash
# Start Ollama service
ollama serve

# In another terminal, pull a recommended model
ollama pull gemma2:12b

# Or try a smaller model for faster processing
ollama pull llama3.2:3b
```

### **Step 3: Clone and Install EasyTrain**

```bash
# Clone the repository
git clone https://github.com/alfredopetruolo/easytrain.git
cd easytrain

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

### **Step 4: Verify Installation**

```bash
# Check system status
python main.py status

# View supported formats
python main.py formats

# Test your model
python main.py test-model gemma2:12b
```

### **Step 5: Convert Your First Document**

```bash
# Basic conversion
python main.py convert path/to/your/document.pdf output.jsonl

# With custom parameters
python main.py convert input.xlsx training_data.jsonl \
  --model gemma2:12b \
  --chunk-size 2000 \
  --verbose
```

---

## 📚 Usage Guide

### **Basic Commands**

```bash
# Convert any supported document
python main.py convert <input_file> <output_file> [options]

# Check system status and available models
python main.py status

# View all supported file formats
python main.py formats

# Test model functionality
python main.py test-model <model_name>
```

### **Advanced Configuration**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Ollama model to use | `gemma2:12b` |
| `--host` | Ollama server URL | `http://localhost:11434` |
| `--chunk-size` | Target chunk size (characters) | `2000` |
| `--overlap-size` | Overlap between chunks | `200` |
| `--max-text-length` | Maximum text per chunk | `4000` |
| `--format` | Output format (json/jsonl) | `jsonl` |
| `--timeout` | Request timeout (seconds) | `3600` |
| `--verbose` | Enable detailed logging | `False` |

### **Example Workflows**

<details>
<summary><strong>📊 Processing Excel Data</strong></summary>

```bash
# Convert financial data to trading knowledge
python main.py convert financial_reports.xlsx trading_knowledge.jsonl \
  --model gemma2:12b \
  --chunk-size 1500

# Output: Q&A pairs about financial principles, analysis methods, trading strategies
```
</details>

<details>
<summary><strong>📄 Converting Research Papers (PDF)</strong></summary>

```bash
# Extract research methodologies and insights
python main.py convert research_paper.pdf methodology_dataset.jsonl \
  --model llama3.2:3b \
  --max-text-length 3000

# Output: Knowledge about research methods, experimental design, analysis techniques
```
</details>

<details>
<summary><strong>📋 Processing Training Manuals (Word)</strong></summary>

```bash
# Convert procedures to instructional knowledge
python main.py convert training_manual.docx procedures_knowledge.jsonl \
  --chunk-size 2500 \
  --verbose

# Output: Step-by-step guides, best practices, expert procedures
```
</details>

---

## 📁 Supported File Formats

<div align="center">

| Format | Extensions | Features | Use Cases |
|--------|------------|----------|-----------|
| **📊 Spreadsheets** | `.csv`, `.xlsx`, `.xls` | Multi-sheet support, data relationships | Financial data, surveys, catalogs |
| **📝 Documents** | `.docx`, `.doc` | Paragraph structure, formatting | Manuals, reports, articles |
| **📄 Plain Text** | `.txt` | Simple text processing, paragraph extraction | Notes, logs, raw text data |
| **📑 PDFs** | `.pdf` | Text + table extraction, OCR support | Research papers, forms, mixed content |
| **📽️ Presentations** | `.pptx`, `.ppt` | Slide structure, title-content pairs | Training materials, lectures |

</div>

---

## 🔧 Advanced Features

### **Custom Prompting**

Create specialized prompts for domain-specific knowledge extraction:

```bash
# Use custom prompt file
python main.py convert medical_data.csv medical_knowledge.jsonl \
  --prompt-file prompts/medical_extraction.txt
```

### **Batch Processing**

```bash
# Process multiple files
for file in documents/*.pdf; do
  python main.py convert "$file" "output/$(basename "$file" .pdf).jsonl"
done
```

### **Configuration Files**

Use configuration files for consistent processing:

```python
# config.yaml
ollama:
  model: "gemma2:12b"
  host: "http://localhost:11434"
chunking:
  chunk_size: 2000
  overlap_size: 200
output_format: "jsonl"
```

---

## 📊 Output Format

EasyTrain generates structured JSON optimized for LLM training:

```json
[
  {
    "instruction": "What are the key principles of effective project management?",
    "input": "",
    "output": "Effective project management relies on five key principles: clear scope definition, realistic timeline planning, resource allocation optimization, stakeholder communication, and continuous risk assessment. These principles ensure project success by maintaining focus, managing expectations, and adapting to challenges."
  },
  {
    "instruction": "How should a project manager handle scope creep?",
    "input": "During project execution, stakeholders request additional features",
    "output": "Address scope creep through formal change control processes: document the request, assess impact on timeline/budget/resources, present options to stakeholders, require written approval for changes, and update project documentation. This maintains project integrity while accommodating necessary changes."
  }
]
```

---

## 🛟 Troubleshooting

<details>
<summary><strong>🚫 Common Issues & Solutions</strong></summary>

**Ollama Connection Issues**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

**Model Not Found**
```bash
# List available models
ollama list

# Pull missing model
ollama pull gemma2:12b
```

**Memory Issues**
```bash
# Reduce chunk size for large documents
python main.py convert large_file.pdf output.jsonl --chunk-size 1000

# Use smaller model
python main.py convert file.pdf output.jsonl --model llama3.2:3b
```

**PDF Processing Errors**
```bash
# Install Java (required for table extraction)
# Ubuntu/Debian:
sudo apt install default-jre poppler-utils tesseract-ocr

# macOS:
brew install openjdk poppler tesseract

# Windows (using chocolatey):
choco install openjdk poppler tesseract
```

**OCR Not Working**
```bash
# Check if Tesseract is installed and in PATH
tesseract --version

# Install language packs for better OCR accuracy
# Ubuntu/Debian:
sudo apt install tesseract-ocr-ita tesseract-ocr-eng

# macOS:
brew install tesseract-lang

# Windows:
# Download language packs from GitHub releases
```

**Poppler Not Found**
```bash
# Ubuntu/Debian:
sudo apt install poppler-utils

# macOS:
brew install poppler

# Windows: Ensure poppler bin directory is in PATH
# Example: C:\Program Files\poppler\bin
```

</details>

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### **Development Setup**

```bash
# Clone your fork
git clone https://github.com/yourusername/easytrain.git
cd easytrain

# Install development dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ollama](https://ollama.com)** - Local LLM inference engine
- **[Python Community](https://python.org)** - Amazing libraries and tools
- **Open Source Contributors** - For inspiration and shared knowledge

---

## 📖 Citation

If this project has been useful for your research, please consider citing the following paper:

```bibtex
@article{coppolino2025asset,
  title={Asset Discovery in Critical Infrastructures: An LLM-Based Approach},
  author={Coppolino, Luigi and Iannaccone, Antonio and Nardone, Roberto and Petruolo, Alfredo},
  journal={Electronics},
  volume={14},
  number={16},
  pages={3267},
  year={2025},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

---

<div align="center">

**⭐ Star this repository if EasyTrain helps your research!**

[Report Bug](https://github.com/alfredopetruolo/easytrain/issues) • [Request Feature](https://github.com/alfredopetruolo/easytrain/issues) • [Documentation](https://github.com/alfredopetruolo/easytrain/wiki)

</div>