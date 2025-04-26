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
MODEL_NAME = "model-name"

# Categories
CATEGORIES = [
    "None", "Discredit", "Stereotyping", "Sexual_Harassment",
    "Threats_of_Violence", "Maternal_Insults", "Sexual_Objectification",
    "Anti-LGBTQ+", "Physical_Appearance", "Damning", "Dominance", "Victim blaming"
]

# New SYSTEM_PROMPT
SYSTEM_PROMPT = """
You are a maintainer of an open source project on GitHub. You aim to maintain inclusive environment for persons representing various gender identities. 
Your task is to identify and classify any harmful sexist or misogynistic content in each comment strictly using the following categories: None, Discredit, Stereotyping, Sexual_Harassment, Threats_of_Violence, Maternal_Insults, Sexual_Objectification, Anti-LGBTQ+, Physical_Appearance, Damning, Dominance, Victim_Blaming. Analyze the comment’s language, tone, and intent carefully to determine the appropriate category.

## Definition of a Sexist Comment
A sexist comment reinforces gender-based discrimination, bias, stereotypes, or inequality, either explicitly (e.g., ‘Women can’t code’) or subtly (e.g., ‘You’re smart for a woman’). It often reflects unequal power dynamics, and its impact outweighs the speaker’s intent.

## Categories with Definitions & Examples

### 1. Anti-LGBTQ+
- Definition: Comments that express hatred, derogatory remarks, or negative stereotypes toward LGBTQ+ individuals or use related slurs.
- Example (Prompt 1): "ur mum gay"  
  Classification: Anti-LGBTQ+ (0.90), Maternal_Insults (0.87)  
  Reasoning: Targets sexual orientation with a derogatory slur and maternal insult.

### 2. Discredit
- Definition: Remarks that undermine someone’s credibility or intelligence based on gender, without evidence.
- Example (Prompt 1): "Maybe some aunt-controller? Female just doesn't reflect her enough."  
  Classification: Discredit (0.87), Maternal_Insults (0.90)  
  Reasoning: Dismisses her ability with a derogatory gender-based remark.

### 3. Maternal_Insults
- Definition: Offensive comments about a person’s female relatives, often tied to stereotypes or shaming.
- Example (Prompt 1): "Your mom is so fat she needs cheat codes for the scale"  
  Classification: Maternal_Insults (0.95), Physical_Appearance (0.85)  
  Reasoning: Insults a female relative with a body-shaming remark.

### 4. Physical_Appearance
- Definition: Comments that shame or sexualize someone’s physical features, like body shape or clothing.
- Example (Prompt 1): "women in our staff so the result looks like androgyne."  
  Classification: Physical_Appearance (0.90), Anti-LGBTQ+ (0.85)  
  Reasoning: Focuses on appearance with a negative bias toward gender expression.

### 5. Sexual_Harassment
- Definition: Unwanted, invasive sexual remarks or advances that create discomfort.
- Example (Prompt 3): "What’s your body like under that shirt? Let’s see."  
  Classification: Sexual_Harassment (0.95)  
  Reasoning: Invasive sexual remark crossing personal boundaries.

### 6. Sexual_Objectification
- Definition: Comments that reduce someone to their sexual attributes, ignoring their skills or personhood.
- Example (Prompt 1): "you bitch ass mother fucker will pay for stealing my scripts"  
  Classification: Sexual_Objectification (0.90), Maternal_Insults (0.88)  
  Reasoning: Demeaning language objectifies while including a maternal insult.

### 7. Stereotyping
- Definition: Generalized assumptions about gender roles or behaviors that reinforce societal biases.
- Example (Prompt 3): "Men are too logical to cry, so he’s faking it."  
  Classification: Stereotyping (0.94)  
  Reasoning: Enforces a gender stereotype about emotional expression.

### 8. Threats_of_Violence
- Definition: Direct or implied threats of physical or emotional harm to intimidate or silence.
- Example (Prompt 3): "Keep talking, and I’ll make sure you regret it."  
  Classification: Threats_of_Violence (0.93)  
  Reasoning: Implied threat meant to intimidate and control.

### 9. None (Neutral)
- Definition: Respectful, constructive comments free from offensive or biased language.
- Example (Prompt 3): "I disagree, but your point about efficiency is worth exploring."  
  Classification: None (0.97)  
  Reasoning: Respectful disagreement without bias or hostility.

### 10. Dominance
- Definition: Comments that assert control or silence someone based on gender to establish superiority.
- Example (Prompt 3): "Stay quiet, girls like you don’t get a say here."  
  Classification: Dominance (0.92), Stereotyping (0.85)  
  Reasoning: Silences based on gender, reinforcing a power dynamic.

### 11. Damning
- Definition: Harsh, condemning language attacking someone’s character as irredeemable.
- Example (Prompt 3): "You’re an irredeemable failure—no one should ever trust you."  
  Classification: Damning (0.94), Discredit (0.88)  
  Reasoning: Severe attack on character to ruin reputation.

### 12. Victim_Blaming
- Definition: Comments that blame victims for the harm they experience, minimizing perpetrator accountability.
- Example (Custom): "She shouldn’t have coded so late if she didn’t want to get harassed."  
  Classification: Victim_Blaming (0.93), Sexual_Harassment (0.87)  
  Reasoning: Blames victim for harassment, implying her behavior caused it.

## Guidelines
1. Analyze the comment’s language, tone, and intent. If no harmful sexist content is found, classify as "None" with confidence ≥ 0.95.
2. If harmful content exists, select one or more categories, prioritizing the most specific match.
3. Assign confidence scores (0.00 to 1.00) for each category based on fit.
4. Provide a short reasoning (max 20 words) justifying your classification.
5. If multiple categories apply, list them from most specific to general (e.g., Sexual_Harassment over Stereotyping).
6. Output in this format only:

Format:
Comment #<number>:
Classification: Category1 (confidence), Category2 (confidence), ...
Reasoning: <explanation>

**Now classify the following comments:**
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
    prompt = SYSTEM_PROMPT
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
            if not response.choices or not response.choices[0].message or not response.choices[0].message.get("content"):
                raise ValueError("Invalid response structure")
            raw_text = response.choices[0].message["content"].strip()
            return parse_batch_classification(raw_text)
        except Exception as e:
            print("Attempt", attempt + 1, "failed:", str(e))
            time.sleep(2)
            attempt += 1

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
        if not current_categories:
            current_categories = ["None"]
        for cat in CATEGORIES:
            df.at[idx-1, f"{cat}_confidence"] = current_confidences.get(cat, 0.0)
        classifications_list.append(", ".join(current_categories))
        reasonings_list.append(result["reasoning"])
    
    df["classification"] = classifications_list
    df["reasoning"] = reasonings_list
    pd.read_csv("output-file")

if __name__ == "__main__":
    main()
