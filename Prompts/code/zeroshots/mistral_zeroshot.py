import os
import re
import pandas as pd
from dotenv import load_dotenv
import together

# Load environment variables and set API key
load_dotenv()
together.api_key = os.getenv("API_KEY")

# Model name
MODEL_NAME = "model-name"
CATEGORIES = [
    "None", 
    "Discredit", 
    "Stereotyping", 
    "Sexual_Harassment",
    "Threats_of_Violence", 
    "Maternal_Insults", 
    "Sexual_Objectification",
    "Anti-LGBTQ+", 
    "Physical_Appearance", 
    "Damning", 
    "Dominance",
    "Victim blaming"
]

# Category definitions for clarity
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
    "None": "No Sexual remark in the comment",
    "Dominance": "To preserve male control / interest and to exclude women from conversation.",
    "Victim blaming": "Blaming the victims for the problems they are facing."
}


ZERO_SHOT_PROMPT = f"""
You are a maintainer of an open source project on GitHub. You aim to maintain inclusive environment for persons representing various gender identities. 
Your task is to identify and classify any harmful sexist or misogynistic content in each comment using only the following categories: {", ".join(CATEGORIES)}.


Category Definitions:
{chr(10).join([f"- {cat}: {definition}" for cat, definition in CATEGORY_DEFINITIONS.items()])}

Instructions:
1. Carefully examine each comment.
2. If no harmful sexist content is found, classify it as "None" with a confidence score ≥ 0.95.
3. If harmful content is detected, select the most appropriate one or more categories.
4. Assign a confidence score (between 0.00 and 1.00) for each category.
5. Provide a short reasoning (≤ 20 words) to justify your classification.
6. Return your response in the exact format shown below without any extra text or commentary.

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
    Sends a batch of comments to the LLM and returns a dictionary of classification results.
    Each comment is numbered so that the output can be parsed accordingly.
    """
    prompt = ZERO_SHOT_PROMPT
    for i, comment in enumerate(comments, 1):
        prompt += f'\nComment #{i}: "{comment}"'
    prompt += "\n\nOutput:"
    
    try:
        response = together.Complete.create(
            prompt=prompt,
            model=MODEL_NAME,
            max_tokens=1000,  # Increased to allow for longer responses
            temperature=0.1,
            top_p=0.9
        )
        raw_text = response['choices'][0]['text'].strip()
        
        return parse_batch_classification(raw_text)
    except Exception as e:
        print(f"Error in LLM call: {e}")
        results = {}
        for i in range(1, len(comments) + 1):
            results[i] = {"classification": [{"category": "None", "confidence": 0.50}],
                          "reasoning": f"Error: {str(e)}"}
        return results

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
    
    df = pd.read_csv("input-file")
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
        for cat in CATEGORIES:
            df.at[idx-1, f"{cat}_confidence"] = current_confidences.get(cat, 0.0)
        classifications_list.append(", ".join(current_categories) if current_categories else "None")
        reasonings_list.append(result["reasoning"])
    
    df["classification"] = classifications_list
    df["reasoning"] = reasonings_list
    df.to_csv("output-file", index=False)

if __name__ == "__main__":
    main()
