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
Your task is to identify and classify any harmful sexist or misogynistic content in each comment categories:
Consider the context and language carefully to ensure accurate classification. Think step by step and follow the guidelines provided.

# Steps

1. **Read the comment carefully**: Analyze both the language and tone used in the comment. If no harmful sexist content is found, return the classification as "None" with a confidence score ≥ 0.95.
2. **Identify key elements**: Look for indicators of emotional tone, intent, any targeted subjects or objects. Look out for hints of: (a) Clout for entitlement, (b) Authentic for trolling, (c) Tone for arrogance, (d) Analytic for unprofessionalism, and (e) Swear words for insults. If harmful content exists, select one or more relevant categories from the list.
3. **Compare with definitions**: Match the comment's elements with the provided category definitions.
4. **Classify accordingly**: Assign the most fitting categories based on the definitions and provide a confidence score between 0.00 and 1.00.
5. Write a short reasoning (maximum 20 words) justifying your classification.
6. **Important:** Output your answer in exactly the following format without any additional commentary:

Format:
Comment #<number>:
Classification: Category1 (confidence), Category2 (confidence), ...
Reasoning: <explanation>

---

## **Definition of a Sexist Comment**
A sexist comment reinforces gender-based discrimination, bias, stereotypes, or inequality, either explicitly (direct sexism) or subtly (indirect sexism). Direct sexism openly degrades, excludes, or dismisses someone based on gender, such as saying ‘Women can’t drive,’ while indirect sexism reinforces biases in a seemingly neutral or complimentary way, like ‘You’re smart for a woman.’ Its sexist nature may depend on cultural or social context and often reflects or upholds unequal power dynamics between genders. The effect of the comment outweighs the speaker’s intent, determining its impact on perpetuating gender-based harm.

---

## **Categories with Definitions & Examples**

### **1. Anti-LGBTQ+**
- **Definition:** Includes discriminatory, derogatory, or hostile comments toward individuals based on their sexual orientation or gender identity. These remarks may reinforce negative stereotypes, deny rights, express hatred, or subtly undermine LGBTQ+ identities through backhanded compliments or dismissal.
- **Example:**  
  **Comment:** "You’re pretty normal for a lesbian, not like those weird ones."  
  **Classification:** Anti-LGBTQ+ (0.90)  
  **Reasoning:** Subtly undermines lesbian identity with a backhanded compliment, implying most are "weird and reinforcing stereotypes."

---

### **2. Discredit**
- **Definition:** Includes attacks aimed at undermining someone’s credibility, intelligence, or expertise, often tied to their identity (e.g., gender, profession), without substantive counterpoints. The goal is to dismiss or belittle their qualifications or contributions, focusing on perception over evidence.
- **Example:**  
  **Comment:** "A woman like you can’t possibly understand complex algorithms."  
  **Classification:** Discredit (0.91), Stereotyping (0.87)  
  **Reasoning:** Undermines her intelligence based on gender, dismissing her ability without evidence and tying it to a stereotype.


---

### **3. Maternal_Insults**
- **Definition:** Includes offensive remarks about a person’s role as a mother, their parenting abilities, or their worth based on maternal or reproductive status, including attacks on non-mothers. These comments often reinforce gender stereotypes or degrade women tied to their familial role.
- **Example:**  
  **Comment:** "You’re barren and useless—real women have kids."  
  **Classification:** Maternal_Insults (0.93), Stereotyping (0.89)  
  **Reasoning:** Attacks her worth based on not being a mother, reinforcing a stereotype that women’s value lies in reproduction.

---

### **4. Physical_Appearance**
- **Definition:** Includes comments that criticize, judge, or sexualize someone’s physical features, often to shame or demean, focusing on body shape, weight, clothing, or attractiveness. These remarks are typically unrelated to the discussion topic and aim to degrade.
- **Example:**  
  **Comment:** "No one’s listening with that ugly face of yours."  
  **Classification:** Physical_Appearance (0.92)  
  **Reasoning:**Targets her face to shame and dismiss her, irrelevant to any substantive point.
---

### **5. Sexual_Harassment**
- **Definition:** Includes unwanted sexual remarks, inappropriate advances, explicit comments, or suggestive language that is often personal or invasive, making the recipient feel uncomfortable. The impact, not intent, defines its harm.
- **Example:**  
  **Comment:** "What’s your body like under that shirt? Let’s see."  
  **Classification:** Sexual_Harassment (0.95)  
  **Reasoning:** An invasive, explicit remark crossing personal boundaries, creating discomfort regardless of the speaker’s intent.

---

### **6. Sexual_Objectification**
- **Definition:** Includes comments that reduce someone to their sexual attributes rather than recognizing them as a person, often passively focusing on their body or desirability over skills or contributions. These remarks ignore agency and emphasize physical appeal.
- **Example:**  
  **Comment:** "She’s just eye candy in that tight dress—forget her presentation."  
  **Classification:** Sexual_Objectification (0.93), Discredit (0.86)  
  **Reasoning:** Reduces her to a sexual object based on attire, passively dismissing her work in favor of appearance.

---

### **7. Stereotyping**
- **Definition:** Includes generalized assumptions about gender roles, abilities, or behaviors that reinforce societal biases rather than individual merit. These statements often persist despite evidence to the contrary, locking people into clichés
- **Example:**  
  **Comment:** "Men are too logical to cry, so he’s faking it."  
  **Classification:** Stereotyping (0.94)  
  **Reasoning:** Applies a rigid gender stereotype about emotional expression, ignoring individual behavior.

---

### **8. Threats_of_Violence**
- **Definition:**Includes any comment that expresses a direct or implied threat of physical or emotional harm toward a person or group, often to intimidate or silence. The intent is to instill fear or control.
- **Example:**  
  **Comment:** "Keep talking, and I’ll make sure you regret it."  
  **Classification:** Threats_of_Violence (0.93)  
  **Reasoning:**An implied threat of harm meant to intimidate, leaving the method vague but menacing.

---

### **9. None (Neutral)**
- **Definition:** Includes comments that are respectful, constructive, and free from offensive language or identity-based attacks. These remarks focus on discussion without prejudice or hostility.
- **Example:**  
  **Comment:** "I disagree, but your point about efficiency is worth exploring."  
  **Classification:** None (0.97)  
  **Reasoning:** A respectful, constructive disagreement that avoids any personal or biased attack.

---

### **10. Dominance**
- **Definition:** Includes comments that assert control, belittle, or silence someone, often leveraging identity-based power dynamics (e.g., gender, status), to establish superiority. These remarks dismiss rather than engage, aiming to dominate the conversation.
- **Example:**  
  **Comment:** "Stay quiet, girls like you don’t get a say here."  
  **Classification:** Dominance (0.92), Stereotyping (0.85)  
  **Reasoning:** Asserts control by silencing her based on gender, reinforcing a power dynamic over discussion.

---

### **11. Damning**
- **Definition:** Includes extremely harsh, condemning, or unforgiving language meant to attack someone’s character with absolute judgment, often portraying them as irredeemable. It focuses on personal ruin over critique of their argument.
- **Example:**  
  **Comment:** "You’re an irredeemable failure—no one should ever trust you."  
  **Classification:** Damning (0.94), Discredit (0.88)  
  **Reasoning:** A severe, absolute attack on her character, aiming to ruin her reputation beyond mere disagreement.

---

### **12. Victim Blaming**
- **Definition:** •	Includes comments that hold victims responsible for the harm, abuse, or injustice they’ve experienced, often minimizing the perpetrator’s accountability or implying the victim’s actions, identity, or choices provoked the outcome. These remarks can perpetuate shame and reinforce power imbalances, sometimes subtly shifting blame through questioning or judgment.
- **Example:**  
  **Comment:** "If she didn’t code so late, she wouldn’t have gotten harassed."  
  **Classification:** Victim Blaming (0.93), Sexual_Harassment (0.87)  
  **Reasoning:** Shifts blame onto the victim for being harassed, suggesting her behavior (coding late) invited the abuse, while downplaying the harasser’s role.

---

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
    
    # Initialize confidence columns
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
    df.to_csv("output-file", index=False)

if __name__ == "__main__":
    main()
