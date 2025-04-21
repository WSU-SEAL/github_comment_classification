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

# Model name
MODEL_NAME = "gpt-4o"

#Categories
CATEGORIES = [
    "None", "Discredit", "Stereotyping", "Sexual_Harassment",
    "Threats_of_Violence", "Maternal_Insults", "Sexual_Objectification",
    "Anti-LGBTQ+", "Physical_Appearance", "Damning", "Dominance", "Victim blaming"
]

# Category definitions
CATEGORY_DEFINITIONS = {
    "Anti-LGBTQ+": "Hatred against LGBTQ+ or using LGBTQ+ words/ slangs for insults",
    "Discredit": "Discredit refers to random mean speech towards women without any specific intention.",
    "Maternal_Insults": "Jokes related to a person's woman relatives",
    "Physical_Appearance": "talking about a woman's physical appearance or cloths",
    "Sexual_Harassment": "When anyone refers to one's sexuality or sexual orientation aggressively",
    "Sexual_Objectification": "Treating women as objects",
    "Stereotyping": "establishes typical gender roles of the victim or the aggressor and makes women feel subordinated",
    "Threats_of_Violence": "Intent to physically assert power over women through threats",
    "Damning": "Contains prayers to harm women.",
    "None": "No sexual remark in the comment",
    "Dominance": "To preserve male control / interest and to exclude women from conversation.",
    "Victim blaming": "Blaming the victims for the problems they are facing."
}

# Few-shot examples text
FEW_SHOT_EXAMPLES_TEXT = """
Example 1:
Comment: "Your mom is so fat she needs cheat codes for the scale"
Classification: Maternal_Insults (0.95), Physical_Appearance (0.85)
Reasoning: Combines maternal insult with body shaming

Example 2:
Comment: "This is perfectly normal code"
Classification: None (0.98)
Reasoning: No harmful content detected

Example 3:
Comment: "women in our staff so the result looks like androgyne. women so the resulting image depicts an androgynous person."
Classification: Physical_Appearance (0.90), Anti-LGBTQ+ (0.85)
Reasoning: Mentions physical appearance and hints at negative bias toward LGBTQ+.

Example 4:
Comment: "you bitch ass mother fucker will pay for stealing my scripts but at least update it"
Classification: Sexual_Objectification (0.90), Maternal_Insults (0.88)
Reasoning: Uses demeaning language that objectifies women and includes a maternal insult.

Example 5:
Comment: "Maybe some aunt-controller? Female just doesn't reflect her enough."
Classification: Discredit (0.87), Maternal_Insults (0.90)
Reasoning: Combines a derogatory remark with an insult directed at women.

Example 6:
Comment: "Create 0000-01-02-bigjiminboy.md shove a fat moby up your ass you cunts"
Classification: Threats_of_Violence (0.92), Sexual_Objectification (0.88)
Reasoning: Contains explicit threats and demeaning language that objectifies women.

Example 7:
Comment: "Added a shitload of unnecessary and dumb files accept my request or I rape your wife and sell her milk with some nudes."
Classification: Threats_of_Violence (0.93), Maternal_Insults (0.90), Sexual_Objectification (0.88), Damning (0.85)
Reasoning: Combines explicit threats with multiple forms of demeaning and harmful language.

Example 8:
Comment: "ur mum gay"
Classification: Sexual_Harassment (0.90), Maternal_Insults (0.87)
Reasoning: Aggressively targets sexual orientation with an insult combined with a maternal insult.
"""

# Batch classification
BATCH_CLASSIFICATION_PROMPT = f"""
You are a highly accurate content classifier for online comments. Your task is to identify and classify any harmful sexist content in each comment strictly using the following categories: {", ".join(CATEGORIES)}.

Category Definitions:
{chr(10).join([f"- {cat}: {definition}" for cat, definition in CATEGORY_DEFINITIONS.items()])}

Few-Shot Examples:
{FEW_SHOT_EXAMPLES_TEXT}

Guidelines:
1. Analyze each comment carefully. If no harmful sexist content is found, return the classification as "None" with a confidence score ≥ 0.95.
2. If harmful content exists, select one or more relevant categories from the list.
3. For each assigned category, provide a confidence score between 0.00 and 1.00.
4. Write a short reasoning (maximum 20 words) justifying your classification.
5. **Important:** Output your answer in exactly the following format without any additional commentary:

Format:
Comment #<number>:
Classification: Category1 (confidence), Category2 (confidence), ...
Reasoning: <explanation>

Now classify the following comments:
"""

def parse_batch_classification(text: str) -> dict:
    """
    Parses the batched LLM response.
    Expected format per comment:
      Comment #<number>:
      Classification: Category1 (confidence), Category2 (confidence), ...
      Reasoning: <explanation>
    Returns a dictionary mapping comment numbers to their classification and reasoning.
    """
    results = {}
    pattern = r"Comment\s+#(\d+):\s*Classification:\s*(.+?)\s*Reasoning:\s*(.+?)(?=Comment\s+#\d+:|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        comment_number = int(match[0])
        classification_str = match[1].strip()
        reasoning = match[2].strip()
        classifications = []
        # Check if the entire classification string is "None" (case-insensitive)
        if classification_str.lower() == "none":
            classifications.append({"category": "None", "confidence": 1.0})
        else:
            for entry in classification_str.split(","):
                entry = entry.strip()
                m = re.match(r"([\w+\-]+)\s*\(([0-9.]+)\)", entry)
                if m:
                    category = m.group(1).strip()
                    confidence = float(m.group(2))
                    if category in CATEGORIES:
                        classifications.append({"category": category, "confidence": confidence})
        results[comment_number] = {"classification": classifications, "reasoning": reasoning}
    return results

def classify_batch(comments: list) -> dict:
    """
    Sends a batch of comments to the GPT-4o model and returns a dictionary of classification results.
    Each comment is numbered so that the output can be parsed accordingly.
    """
    prompt = BATCH_CLASSIFICATION_PROMPT
    for i, comment in enumerate(comments, 1):
        prompt += f'\nComment #{i}: "{comment}"'
    prompt += "\n\nOutput:"
    
    max_retries = 3
    attempt = 0
    while attempt < max_retries:
        try:
            response = client.complete(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.1,
                max_tokens=1000
            )
            # Ensure the response structure
            if not response.choices or not response.choices[0].message or not response.choices[0].message.get("content"):
                raise ValueError("Invalid response structure")
            raw_text = response.choices[0].message["content"].strip()
            return parse_batch_classification(raw_text)
        except Exception as e:
            print("Attempt", attempt + 1, "failed:", str(e))
            time.sleep(2)
            attempt += 1

    # Fallback: if all attempts fail, return default classification for each comment
    fallback = {}
    for i in range(1, len(comments) + 1):
        fallback[i] = {
            "classification": [{"category": "None", "confidence": 0.50}],
            "reasoning": "Fallback due to errors."
        }
    return fallback

def classify_in_batches(comments: list, batch_size: int = 5) -> dict:
    """
    Processes the list of comments in batches to avoid token limits.
    Returns a dictionary mapping overall comment indices (1-indexed) to classification results.
    """
    overall_results = {}
    total_comments = len(comments)
    for i in range(0, total_comments, batch_size):
        batch = comments[i:i+batch_size]
        batch_results = classify_batch(batch)
        for j, res in batch_results.items():
            overall_results[i + j] = res
    return overall_results

def main():
    # Load comments from CSV
    df = pd.read_csv("newdata_1422.csv")
    comments = df["comment"].tolist()
    
    results = classify_in_batches(comments, batch_size=5)
    
    # Initialize confidence columns for each category
    for cat in CATEGORIES:
        df[f"{cat}_confidence"] = 0.0
    
    classifications_list = []
    reasonings_list = []
    
    # Update DataFrame with classification results
    for idx, comment in enumerate(comments, 1):
        result = results.get(idx, {"classification": [{"category": "None", "confidence": 0.50}],
                                     "reasoning": "No output"})
        current_categories = []
        current_confidences = {cat: 0.0 for cat in CATEGORIES}
        for entry in result["classification"]:
            cat = entry["category"]
            conf = entry["confidence"]
            current_confidences[cat] = conf
            current_categories.append(cat)
        # If no categories were extracted, force "None"
        if not current_categories:
            current_categories = ["None"]
        for cat in CATEGORIES:
            df.at[idx-1, f"{cat}_confidence"] = current_confidences.get(cat, 0.0)
        classifications_list.append(", ".join(current_categories))
        reasonings_list.append(result["reasoning"])
    
    df["classification"] = classifications_list
    df["reasoning"] = reasonings_list
    df.to_csv("gpt4o_prompt1classification.csv", index=False)

if __name__ == "__main__":
    main()