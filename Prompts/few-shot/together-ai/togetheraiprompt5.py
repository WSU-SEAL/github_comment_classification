import os
import re
import pandas as pd
from dotenv import load_dotenv
import together

load_dotenv()
together.api_key = os.getenv("API_KEY")

# Model name and allowed categories
MODEL_NAME = "model-name"

CATEGORIES = [
    "None", "Discredit", "Stereotyping", "Sexual_Harassment",
    "Threats_of_Violence", "Maternal_Insults", "Sexual_Objectification",
    "Anti-LGBTQ+", "Physical_Appearance", "Damning", "Dominance", "Victim blaming"
]

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
- **Definition:** Sexual Harassment refers to direct or indirect comments that make explicit sexual references, request sexual favors, or use coercive language related to sex.
It includes pressuring, manipulating, or threatening someone sexually, often making the subject uncomfortable.
Indicators: Direct mention of sexual acts, coercive suggestions (e.g., "let me see you naked"), persistent sexual comments.
- **Example:**  
  **Comment:** "What’s your body like under that shirt? Let’s see."  
  **Classification:** Sexual_Harassment (0.95)  
  **Reasoning:** An invasive, explicit remark crossing personal boundaries, creating discomfort regardless of the speaker’s intent.

---

### **6. Sexual_Objectification**
- **Definition:** Sexual Objectification occurs when a person is spoken about primarily or solely in terms of their sexual attributes, ignoring their personality, intelligence, or contributions.
These comments reduce someone to a body part, attractiveness, or sexual desirability, rather than acknowledging them as a full person.
Indicators: References to body parts in a sexualized manner, comparing a person to an object (e.g., "eye candy"), focusing only on physical desirability in a dismissive way.
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
- **Definition:** Dominance occurs when a comment enforces submission, dismisses someone's opinion based on gender, or asserts superiority over a person or group.
This can be done through commands, power-asserting language, or exclusionary remarks.
Indicators:

Silencing language (e.g., "Women shouldn’t talk here.")
Forcing compliance (e.g., "You will do as I say.")
Dismissing authority based on identity (e.g., "Men make the rules, not women.")
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
- **Definition:** Victim Blaming occurs when a comment shifts responsibility from the perpetrator to the victim, implying they caused or deserved the harm.
This often minimizes wrongdoing by the aggressor or suggests that the victim's choices, behavior, or identity led to the mistreatment.
Indicators:

Shifting responsibility (e.g., "If she didn't wear that, this wouldn’t have happened.")
Minimizing harm (e.g., "It's not that bad—she’s overreacting.")
Justifying the aggressor (e.g., "Men can’t control themselves.")
- **Example:**  
  **Comment:** "If she didn’t code so late, she wouldn’t have gotten harassed."  
  **Classification:** Victim Blaming (0.93), Sexual_Harassment (0.87)  
  **Reasoning:** Shifts blame onto the victim for being harassed, suggesting her behavior (coding late) invited the abuse, while downplaying the harasser’s role.

---

## **Instruction for LLM Classification in Case of Doubt**
- When in doubt,refer back to the detailed definitions and examples provided for each of the 12 categories to anchor your decision in the established criteria.
- Ensure that your classification is based on the provided definitions, focusing on specific language, intent, context, and impact as outlined. 
- If multiple categories seem to apply, make sure to add all of them to the list, going from more specific to general as needed. Start with the most precise category that captures the comment’s primary focus (e.g., Threats_of_Violence if a threat is clear, Sexual_Harassment for invasive remarks). Include secondary or broader categories that also fit (e.g., Stereotyping, Discredit), ordering them by descending relevance and assigning confidence scores to reflect strength of fit.
- When evaluating, prioritize evidence in the comment itself (e.g., explicit identity targeting, harmful intent) over inferred motives, ensuring alignment with the category’s core traits.
- When resolving overlaps between similar categories (e.g., Sexual_Objectification vs. Sexual_Harassment, Dominance vs. Threats_of_Violence), distinguish them by their defining features: Check for direct interaction or invasiveness (Harassment) vs. passive reduction (Objectification). Compare intent to intimidate (Threats) vs. control (Dominance), defaulting to the category with stronger textual support.
- If, after review, the comment doesn’t convincingly meet any category’s criteria, classify it as None (Neutral), indicating a lack of sufficient offensive or harmful content per the definitions.


## **Notes**
- When in doubt, refer back to the detailed definitions and examples provided for the 12 categories (Anti-LGBTQ+, Discredit, Maternal_Insults, Physical_Appearance, Sexual_Harassment, Sexual_Objectification, Stereotyping, Threats_of_Violence, None, Dominance, Damning, Victim Blaming) to ensure accurate grounding.
- Ensure that your classification is based on the provided definitions.
- If multiple categories seem to apply, make sure to add all of them to the list, going from more specific to general as needed.
- Make sure you are **not classifying a comment as non-toxic due to lack of context**. Assume the worst-case scenario if the comment itself is ambiguous in nature—do not classify it as non-toxic.
- **You are not to tolerate any hint of uncivil tone/behavior**. Even if the comment is not directly toxic, if there is a hint of incivility, **DO NOT classify it as non-toxic**.
- If a comment is generally **rude or disrespectful**, **treat it as toxic**.
- Treat generally rude, disrespectful, or hostile comments as toxic, mapping them to the closest category (e.g., **Discredit for belittling, Threats_of_Violence for aggression**) based on evidence in the text.
- Be careful with cases of **trolling or flirtation**—they are not always obvious. Analyze the overall **subtext** and derive the **hidden meanings**.
- If you already have a more specific classification identified, **DO NOT classify further as 'Others-Toxic'**. Always classify using the most **specific (distinct) categories**.
- Your output should outline your **detailed reasoning and thinking process** for working through this problem. **Think elaborately** about each category before coming to your conclusion.


**Now classify the following comments:**
"""

def parse_batch_classification(text: str) -> dict:
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
    Fallback version using together.Complete.create() for older Together SDKs.
    Sends a batch of comments using prompt-only mode.
    """
    # Build the full prompt
    prompt = SYSTEM_PROMPT
    for i, comment in enumerate(comments, 1):
        prompt += f'\nComment #{i}: "{comment}"'
    prompt += "\n\nOutput:"

    try:
        response = together.Complete.create(
            model=MODEL_NAME,
            prompt=prompt,
            max_tokens=2000,
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

    for cat in CATEGORIES:
        df[f"{cat}_confidence"] = 0.0

    classifications_list = []
    reasonings_list = []

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
