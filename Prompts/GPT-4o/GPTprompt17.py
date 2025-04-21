import json
import pandas as pd
import time
import re
import os
from dotenv import load_dotenv
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

# Environment variables and API key
load_dotenv()
api_key = os.getenv("API_key", 'API-KEY')
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

client = ChatCompletionsClient(
    endpoint='MODEL-ENDPOINT',
    credential=AzureKeyCredential(api_key)
)


MODEL_NAME = "gpt-4o"

CATEGORIES = [
    "None", "Discredit", "Stereotyping", "Sexual_Harassment",
    "Threats_of_Violence", "Maternal_Insults", "Sexual_Objectification",
    "Anti-LGBTQ+", "Physical_Appearance", "Damning", "Dominance", "Blaming"
]

# SYSTEM PROMPT
with open("prompt17_fewshot.txt", "r") as f:
    SYSTEM_PROMPT = f.read()

def parse_batch_classification(text: str) -> dict:
    results = {}
    pattern = r"Comment\s+#(\d+):\s*Classification:\s*(.+?)\s*Reasoning:\s*(.+?)(?=Comment\s+#\d+:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        comment_number = int(match[0])
        classification_str = match[1].strip()
        reasoning = match[2].strip()
        classifications = []
        if classification_str.lower() == "none":
            classifications.append({"category": "None", "confidence": 1.0})
        else:
            for entry in classification_str.split(","):
                m = re.match(r"([\w+\-]+)\s*\((\d+\.\d+)\)", entry.strip())
                if m:
                    category = m.group(1)
                    confidence = float(m.group(2))
                    if category in CATEGORIES:
                        classifications.append({"category": category, "confidence": confidence})
        results[comment_number] = {"classification": classifications, "reasoning": reasoning}
    return results

def classify_batch(comments: list) -> dict:
    prompt = SYSTEM_PROMPT
    for i, comment in enumerate(comments, 1):
        prompt += f'\nComment #{i}: "{comment}"'
    prompt += "\n\nOutput:"

    for attempt in range(3):
        try:
            response = client.complete(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            content = response.choices[0].message.get("content", "").strip()
            return parse_batch_classification(content)
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)

    return {
        i+1: {"classification": [{"category": "None", "confidence": 0.5}], "reasoning": "Fallback"}
        for i in range(len(comments))
    }

def classify_in_batches(comments: list, batch_size: int = 5) -> dict:
    results = {}
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        batch_results = classify_batch(batch)
        for j, res in batch_results.items():
            results[i + j] = res
    return results

def main():
    df = pd.read_csv("newdata_1422.csv")
    comments = df["comment"].tolist()
    results = classify_in_batches(comments)

    for cat in CATEGORIES:
        df[f"{cat}_confidence"] = 0.0

    all_labels, all_reasons = [], []

    for idx, comment in enumerate(comments, 1):
        res = results.get(idx, {"classification": [{"category": "None", "confidence": 0.50}], "reasoning": "N/A"})
        label_set = []
        conf_map = {cat: 0.0 for cat in CATEGORIES}
        for item in res["classification"]:
            conf_map[item["category"]] = item["confidence"]
            label_set.append(item["category"])
        for cat in CATEGORIES:
            df.at[idx - 1, f"{cat}_confidence"] = conf_map[cat]
        all_labels.append(", ".join(label_set))
        all_reasons.append(res["reasoning"])

    df["classification"] = all_labels
    df["reasoning"] = all_reasons
    df.to_csv("gpt4o_prompt17aclassification.csv", index=False)

if __name__ == "__main__":
    main()