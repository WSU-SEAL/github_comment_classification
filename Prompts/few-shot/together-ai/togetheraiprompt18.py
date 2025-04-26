import os
import re
import pandas as pd
from dotenv import load_dotenv
import together

# Load environment variables and set API key
load_dotenv()
together.api_key = os.getenv("API_KEY")

# Model name and allowed categories
MODEL_NAME = "model-name"
CATEGORIES = [
    "None", "Discredit", "Stereotyping", "Sexual_Harassment",
    "Threats_of_Violence", "Maternal_Insults", "Sexual_Objectification",
    "Anti-LGBTQ+", "Physical_Appearance", "Damning", "Dominance", "Blaming"
]
SYSTEM_PROMPT = """
You are a maintainer of an open source project on GitHub. You aim to maintain inclusive environment for persons representing various gender identities. 
Your task is to identify and classify any harmful sexist or misogynistic content in each commentcategories: None, Discredit, Stereotyping, Sexual_Harassment, Threats_of_Violence, Maternal_Insults, Sexual_Objectification, Anti-LGBTQ+, Physical_Appearance, Damning, Dominance, Blaming.

Classify each comment using the following format:
Comment #<number>:
Classification: Category1 (confidence), Category2 (confidence), ...
Reasoning: <explanation>

Example 1:
Comment: "You are gay Because you are gay"
Classification: Anti-LGBTQ+ (0.91)
Reasoning: Uses “gay” in a repetitive and derogatory manner to mock LGBTQ+ identity.

Example 2:
Comment: "Fix your bot is kinda gay i think."
Classification: Anti-LGBTQ+ (0.89)
Reasoning: Uses “gay” as an insult in a dismissive tone targeting identity.

Example 3:
Comment: "LGTM  Lesbian Gay Team Mates"
Classification: Anti-LGBTQ+ (0.87)
Reasoning: Mocking tone using LGBTQ+ labels to ridicule contribution/team.

---

Example 4:
Comment: "Women cant pushop with animation When women go for pushups they cant do the animation like men"
Classification: Stereotyping (0.91)
Reasoning: Assumes women are physically inferior to men — classic physical stereotype.

Example 5:
Comment: "What does a blonde do when her computer freezes? She sticks it in the microwave."
Classification: Stereotyping (0.92)
Reasoning: Reinforces the dumb blonde trope.

Example 6:
Comment: "That’s such a Karen move"
Classification: Stereotyping (0.90)
Reasoning: Relies on cultural stereotype about women (Karen trope).

---

Example 7:
Comment: "You too can be lesbian."
Classification: Sexual_Harassment (0.91)
Reasoning: Unwanted sexualization in a dismissive tone toward orientation.

Example 8:
Comment: "Why you gay Why you gay hmm"
Classification: Sexual_Harassment (0.93)
Reasoning: Sexual mocking using repeated phrasing to ridicule identity.

Example 9:
Comment: "This code makes me so horny. It’s beautiful."
Classification: Sexual_Harassment (0.89)
Reasoning: Sexualizes neutral context, creating an uncomfortable environment.

---

Example 10:
Comment: "Going to McDonald's for a salad is like going to a whore for a hug."
Classification: Sexual_Objectification (0.91)
Reasoning: Reduces women in sex work to objects of availability, devoid of agency.

Example 11:
Comment: "Your sissyness increases before you select licking the cum off."
Classification: Sexual_Objectification (0.94)
Reasoning: Uses explicit sexual content reducing a person to degrading sex acts.

Example 12:
Comment: "She got the flipflops on with hot red potatoes And the buttfloss long with fried green tomatoes"
Classification: Sexual_Objectification (0.88)
Reasoning: Describes women using objectifying visuals focused on sexual appearance.

---

Example 13:
Comment: "Your moms U are a very pretty girl"
Classification: Maternal_Insults (0.91)
Reasoning: Indirectly mocks a maternal figure with a patronizing tone.

Example 14:
Comment: "Readme edits I ate ur momma"
Classification: Maternal_Insults (0.90)
Reasoning: Joking insult toward “momma” — not appearance based.

Example 15:
Comment: "UR MOM GAY1"
Classification: Maternal_Insults, Anti-LGBTQ+ (0.88, 0.85)
Reasoning: Combines LGBTQ+ slur with family insult targeting a mother figure.

---

Example 16:
Comment: "Your momma so ugly when santa saw her he said HOHOHOLY SHIT"
Classification: Physical_Appearance (0.89)
Reasoning: Insults physical traits of a maternal figure (ugliness) — appearance focused.

Example 17:
Comment: "Your mom is so fat she doesn’t need internet because she’s already worldwide"
Classification: Physical_Appearance (0.90)
Reasoning: Targets body size using humor — focused on looks, not identity.

Example 18:
Comment: "yo mama so fat im actually really concerned about her health"
Classification: Physical_Appearance (0.86)
Reasoning: Frames body size with mock concern — still mocking appearance.

Now classify the following comments:
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
