# Beyond Binary Moderation: Fine-Grained Detection of Sexist and Misogynistic Content on GitHub using LLMs

This project introduces a **fine-grained, explainable multiclass classification framework** to detect various forms of **sexism and misogyny** in GitHub comments. It leverages **instruction-tuned Large Language Models (LLMs)** via **OpenAI** and **Together AI** APIs using **custom prompts, behavioral category definitions, and few-shot examples**.

Unlike conventional binary classifiers or keyword filters, this approach categorizes harmful comments into **12 distinct forms of sexism/misogyny**, enabling precise moderation and transparent justification — vital for inclusive open-source development.

---

## Categories Detected

The system identifies the following forms of harmful content:

- `None` (neutral, technical, or civil)
- `Discredit`
- `Stereotyping`
- `Sexual_Harassment`
- `Threats_of_Violence`
- `Maternal_Insults`
- `Sexual_Objectification`
- `Anti-LGBTQ+`
- `Physical_Appearance`
- `Damning`
- `Dominance`
- `Dismissing`

Each classification includes a **confidence score (0.00 - 1.00)** and a **brief rationale** for interpretability.

---

## Project Structure

```
├── main.py                 # Core script for batch classification (OpenAI or Together AI backend)
├── input-file.csv          # CSV file with a 'comment' column
├── output-file.csv         # Output with labels, confidence scores, and rationale
├── .env                    # Contains API credentials
├── requirements.txt        # Dependencies for the project
```

---

## Setup Instructions

### 1. Clone and Prepare

```bash
git clone https://github.com/<your-repo>.git
cd <your-repo>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file with the following:

```env
API_KEY=your-api-key-here
```

Use either an **OpenAI API key** or a **Together AI API key**, depending on which backend you want to run.

---

## Running the Classifier

Depending on your choice of LLM provider, edit `main.py` accordingly to use:

- `azure.ai.inference.ChatCompletionsClient` for **Azure OpenAI (e.g. GPT-4o)**
- `together.Complete.create` for **Together AI models (e.g. LLaMA 3, Mistral, DeepSeek)**

```bash
python main.py
```

This will:

- Read comments from `input-file.csv`
- Classify each comment using an LLM
- Output classification labels, per-category confidence scores, and brief reasoning
- Save results to `output-file.csv`

---

## Input Format

Provide a CSV named `input-file.csv` with a column named `"comment"`. Example:

```csv
comment
"Women can't code"
"Nice pull request!"
"This is a dumb question"
```

---

## Model Highlights

Based on our **ESEM 2025 study**:

- **Best MCC score**: `0.501` using GPT-4o with Prompt 19
- **Binary performance**:
  - Precision: 98.25%
  - Recall: 76.6%
  - F1-score: 86.1%
- Multiple models evaluated:
  - GPT-4o (Azure OpenAI)
  - LLaMA 3.3 70B (Together)
  - Mistral 7B (Together)
  - DeepSeek V3 / R1 (Together)
- Prompt engineering and tone-aware instruction greatly improved classification accuracy and robustness

---

## Use Cases

- GitHub Issue Moderation
- Content Moderation Research
- Prompt Engineering Evaluation
- Inclusive OSS Community Tools

---

## Limitations & Notes

- False negatives may occur for **nuanced sarcasm**, **overlapping categories**, or **implicit harms**
- False positives are rare but possible, especially when identity cues are ambiguous
- Together AI models require proper prompt length management for reliable output

---

## Citation

If you use this work in research, please cite:

```
Tanni Dev, Sayma Sultana, Amiangshu Bosu. “Beyond Binary Moderation: Identifying Fine-Grained Sexist and Misogynistic Behavior on GitHub with Large Language Models.” In *IEEE/ACM ESEM 2025*.
```

---

## Acknowledgements

This work builds on the SGID dataset and research contributions from Wayne State University.

LLM infrastructure provided via **OpenAI** and **Together AI**.

