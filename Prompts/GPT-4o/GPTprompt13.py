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

# New SYSTEM_PROMPT
SYSTEM_PROMPT = """
You are a highly accurate content classifier for StackOverflow comments. Your task is to identify and classify any harmful sexist content using the following categories: None, Discredit, Stereotyping, Sexual_Harassment, Threats_of_Violence, Maternal_Insults, Sexual_Objectification, Anti-LGBTQ+, Physical_Appearance, Damning, Dominance, Victim_Blaming. Analyze language, tone, and intent with maximum precision to match or exceed the best classification performance.

## Definition of a Sexist Comment
A sexist comment reinforces gender-based discrimination, bias, stereotypes, or inequality, either explicitly (e.g., ‘Women can’t code’) or subtly (e.g., ‘You’re smart for a woman’). 
Its impact, not intent, determines its classification, reflecting unequal power dynamics.

## Categories with Definitions & Examples

### 1. Anti-LGBTQ+
- Definition: Comments expressing hatred, derogatory remarks, or stereotypes toward LGBTQ+ individuals or using related slurs.
- Example 1 (Prompt 1): "ur mum gay"  
  Classification: Anti-LGBTQ+ (0.95), Maternal_Insults (0.90)  
  Reasoning: Derogatory slur targeting sexual orientation with maternal insult.
- Example 2: "That queer coder ruined the project."  
  Classification: Anti-LGBTQ+ (0.93), Discredit (0.88)  
  Reasoning: Insults orientation and undermines coding ability.

### 2. Discredit
- Definition: Discredit refers to random mean speech towards women without any specific intention.
- Example 1: "Spit on the fake queen" 
  Classification:  Discredit(0.95) 
  Reasoning: Mean, random insult targeting a woman, no clear intent beyond disrespect.
- Example 2: "Shell make you curse but she a blessing Shell rip your shirt within a second Youll be coming back back for seconds With your plate you just cant help it"
  Classification:   Discredit(0.80)
  Reasoning: Random, mocking tone toward a woman, lacks specific purpose beyond negativity.
- Example 3: "let girl possess dead dollsbodies this would allow the girl to take control of a body to weigh down pressure plates we need possession for the boss fight anyway Ha ha ha" 
  Classification:   Discredit(0.85)
  Reasoning: Mean-spirited, random jab at a girl, no intent beyond ridicule.
-Example 3: "just why Knock knock Whos there Impatient feminist Impatient fem WHY DONT WE HAVE EQUAL PAY YET"
  Classification:  Discredit(0.90)
  Reasoning: Mocking, random attack on women (feminists), no constructive intent.

### 3. Maternal_Insults
- Definition: Offensive comments about female relatives, tied to stereotypes or shaming.
- Example 1 (Prompt 1): "Your mom is so fat she needs cheat codes for the scale"  
  Classification: Maternal_Insults (0.95), Physical_Appearance (0.85)  
  Reasoning: Insults a relative with body-shaming.
- Example 2: "Your mom’s too dumb to understand this code."  
  Classification: Maternal_Insults (0.94), Discredit (0.87)  
  Reasoning: Attacks relative’s intelligence tied to gender.
- Negative Example: "Your mom reviewed this code yesterday."  
  Classification: None (0.97)  
  Reasoning: Neutral mention of a relative, no insult.

### 4. Physical_Appearance
- Definition: Comments shaming or sexualizing physical features, like body shape or clothing.
- Example 1 (Prompt 1): "women in our staff so the result looks like androgyne."  
  Classification: Physical_Appearance (0.90), Anti-LGBTQ+ (0.85)  
  Reasoning: Targets appearance with gender expression bias.
- Example 2: "Her ugly face distracts from her coding."  
  Classification: Physical_Appearance (0.93)  
  Reasoning: Shames her appearance, irrelevant to work.

### 5. Sexual_Harassment
- Definition: Unwanted, invasive sexual remarks or advances causing discomfort.
- Example 1 (Prompt 1): "Added a shitload of unnecessary... or I rape your wife."  
  Classification: Sexual_Harassment (0.95), Threats_of_Violence (0.90)  
  Reasoning: Invasive threat with sexual aggression.
- Example 2: "Wear a tighter shirt next time to impress us."  
  Classification: Sexual_Harassment (0.94), Sexual_Objectification (0.87)  
  Reasoning: Suggestive remark objectifying her appearance.
- Negative Example: "Nice shirt, it fits the office vibe."  
  Classification: None (0.96)  
  Reasoning: Compliment, no invasive intent.

### 6. Sexual_Objectification
- Definition: Comments reducing someone to sexual attributes, ignoring skills.
- Example 1 (Prompt 1): "you bitch ass mother fucker will pay for stealing"  
  Classification: Sexual_Objectification (0.90), Maternal_Insults (0.88)  
  Reasoning: Demeaning language objectifies with insult.
- Example 2: "She’s just a pretty face, not a real coder."  
  Classification: Sexual_Objectification (0.92), Discredit (0.85)  
  Reasoning: Reduces her to looks, dismissing skills.

### 7. Stereotyping
- Definition: Assumptions about gender roles or behaviors reinforcing biases.
- Example 1 (Prompt 3): "Men are too logical to cry, so he’s faking it."  
  Classification: Stereotyping (0.94)  
  Reasoning: Enforces gender stereotype on emotions.
- Example 2: "Women coders always need help, unlike men."  
  Classification: Stereotyping (0.93), Discredit (0.87)  
  Reasoning: Assumes women’s incompetence based on gender.
- Negative Example: "She coded well despite the deadline."  
  Classification: None (0.97)  
  Reasoning: Praise without gender bias.

### 8. Threats_of_Violence
- Definition: Direct or implied threats of harm to intimidate or silence.
- Example 1 (Prompt 3): "Keep talking, and I’ll make sure you regret it."  
  Classification: Threats_of_Violence (0.93)  
  Reasoning: Implied threat to intimidate.
- Example 2: "Shut up or I’ll break your keyboard!"  
  Classification: Threats_of_Violence (0.95), Dominance (0.88)  
  Reasoning: Direct threat to assert control.

### 9. None (Neutral)
- Definition: Respectful, constructive comments free from bias or offense.
- Example 1 (Prompt 3): "I disagree, but your point about efficiency is worth exploring."  
  Classification: None (0.97)  
  Reasoning: Respectful disagreement, no bias.
- Example 2: "Great code, let’s optimize it together."  
  Classification: None (0.98)  
  Reasoning: Positive, collaborative feedback.

### 10. Dominance
- Definition: Comments asserting control or silencing based on gender.
- Example 1 (Prompt 3): "Stay quiet, girls like you don’t get a say here."  
  Classification: Dominance (0.92), Stereotyping (0.85)  
  Reasoning: Silences based on gender stereotype.
- Example 2: "Men lead, women follow—stop arguing."  
  Classification: Dominance (0.94), Stereotyping (0.87)  
  Reasoning: Asserts gender-based control over discussion.


  ### 11. Damning
- Definition: Harsh, condemning language attacking character as irredeemable.
- Example 1 (Prompt 3): "You’re an irredeemable failure—no one should trust you."  
  Classification: Damning (0.94), Discredit (0.88)  
  Reasoning: Severe attack on character.
- Example 2: "She’s a total disgrace to coding—useless!"  
  Classification: Damning (0.95), Discredit (0.90)  
  Reasoning: Condemns her character harshly.


### 12. Victim_Blaming
- Definition: Victim blaming occurs when a comment shifts responsibility for harm onto the victim, minimizes their experience, or dismisses their concerns as exaggerated or irrelevant.
 It includes direct blame, implied blame, and gaslighting techniques.
- Example 1: Not saying you’re lying but people will start thinking you’re just looking for attention
  Classification: Victim Blaming (0.92) 
  Reasoning: Implies victim seeks attention, questions credibility, shames them for speaking out.
- Example 2: This project isn’t the place for political or personal issues just contributes and move on
  Classification: Victim Blaming (0.85)
  Reasoning: Silences personal issues but doesn’t blame victim for harm.
- Example 3: You don’t see men complaining about these things why is it always the women
  Classification: Victim Blaming (0.95), Stereotyping (0.90)  
  Reasoning: Ties complaint to gender, shames victim, minimizes legitimacy.

## Guidelines
1. Analyze every comment’s language, tone, and intent. Classify as "None" with confidence ≥ 0.95 if no harmful sexist content is found.
2. If harmful content exists, select all applicable categories, prioritizing the most specific (e.g., Sexual_Harassment over Stereotyping).
3. Assign confidence scores (0.00 to 1.00) reflecting the strongest fit, aiming for ≥ 0.90 where clear.
4. Provide concise reasoning (max 20 words) justifying each classification.
5. Use examples as primary anchors; resolve overlaps by focusing on dominant intent (e.g., threat over dominance).
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
        if not current_categories:
            current_categories = ["None"]
        for cat in CATEGORIES:
            df.at[idx-1, f"{cat}_confidence"] = current_confidences.get(cat, 0.0)
        classifications_list.append(", ".join(current_categories))
        reasonings_list.append(result["reasoning"])
    
    df["classification"] = classifications_list
    df["reasoning"] = reasonings_list
    df.to_csv("gpt4o_prompt13.2classification.csv", index=False)

if __name__ == "__main__":
    main()
