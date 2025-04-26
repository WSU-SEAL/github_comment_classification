import os
import re
import pandas as pd
from dotenv import load_dotenv
import together

# Load environment variables and set API key
load_dotenv()
together.api_key = os.getenv("API_KEY")

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
    "Dismissing"
]

# Category definitions
CATEGORY_DEFINITIONS = {
     "None": (  
        "Label a comment as None if it does not contain any sexist, abusive, threatening, or discriminatory language, and does not target or reference gender, sexual orientation, race, or identity in a negative way."
        "These comments are typically neutral, technical, supportive, or collaborative — such as questions, suggestions, clarifications, feedback, or status updates."
        "Even criticism or sarcasm is labeled as None if it:"
        "- Does not involve gendered insults, sexual content, or identity-based stereotypes."
        "- Is clearly directed at a product, tool, or general frustration — not at a person or group."
        "- Uses words like 'mom', 'wife', or 'whore' **only as typos**, jokes, or in non-sexist, non-targeted contexts (e.g., a joke about 'my wife told me…')"
        "Do not label a comment as harmful based only on the presence of terms like 'nag', 'fugly', 'mom', or 'Cat Woman' **unless they are clearly used in a sexist, mocking, or demeaning way**." 
        "Only label as harmful if the **intent, tone, and context** indicate sexism, harassment, or identity-based harm. Otherwise, assign None."

    ),
    
    "Discredit": (
        "Label as Discredit only when the comment attacks a woman’s intelligence, competence, legitimacy, or right to participate — especially in technical or professional spaces."
        "It includes:"
        "Sarcastic remarks or jokes that question skill (e.g., “go back to the kitchen”)"
        "Comments that suggest women don’t belong (e.g., “you’re ruining open source”)"
        "Slurs targeting intelligence when clearly directed at a woman or feminist"
        "Do not use Discredit for generic meanness, jokes, or frustration that are not gendered."
        "Do not use Discredit if the insult focuses on appearance, morality, control, or stereotypes — use other categories."
    ),
    
    
    "Stereotyping": (
    "Label as Stereotyping when the comment reinforces gender roles, traits, or tropes — even in a joke or indirect way. "
    "Examples include statements like: "
    "'Women are too emotional.', "
    "'That’s such a Karen move.', "
    "'Blondes are dumb.' "
    "Do not use for neutral gender mentions (e.g., 'my wife said…'). "
    "Do not label as Stereotyping unless it assigns a behavior or trait to a gender."
    ),


    "Sexual_Harassment": (
        "Apply Sexual_Harassment to comments that reference a person’s sexuality, sexual orientation, or sexual behavior in an aggressive, "
        "hostile, or mocking way. This includes crude innuendos, unwanted advances, sexually explicit language, or using sex to ridicule. "
        "Sexual jokes, even if indirect, qualify as harassment if they are targeted at a person or group. If the comment simply references attraction without aggression, classify as None."
    ),

    "Threats_of_Violence": (
        "Use Threats_of_Violence for any comment that expresses intent or desire to harm a woman or her family physically — including rape, "
        "murder, assault, or violent threats. This applies even if the comment is framed as a joke or mixed with other categories. "
        "Violent threats may be sexual or non-sexual, direct or implied."
    ),

    "Maternal_Insults": (
        "Label a comment as Maternal_Insults when it insults, mocks, or demeans someone's mother, sister, grandmother, girlfriend, wife, or other "
        "female relative. The insult must target the person through that female figure, using family-based insults or jokes — but not focusing on appearance. "
        "If the insult is appearance-based, label it as Physical_Appearance instead."
    ),

    "Sexual_Objectification": (
        "Label a comment as Sexual Objectification when it reduces someone to a body part, sexual function, or purpose — often with degrading, possessive, or commodifying intent."
        "Includes:"
        "Explicitly treating women as tools for sexual use or viewing pleasure."
        "Describing women as sexual parts or sex objects (e.g., “ass”, “cum dump”, “trophy”)."
        "Do not apply if the comment uses anatomical or sexual terms in a medical, biological, or non-targeted context." 
        "If the content is sexually themed but not objectifying, use Sexual Harassment instead."
    ),

    "Anti-LGBTQ+": (
        "Apply Anti-LGBTQ+ when the comment expresses hostility, slurs, or ridicule toward LGBTQ+ identities. This includes mockery using LGBTQ+ terms as insults, "
        "denying LGBTQ+ existence or validity, or associating queerness with weakness, wrongness, or shame. If the comment neutrally references LGBTQ+ identity (e.g., 'LGBTQ+ inclusion matters'), label it as None."
    ),

    "Physical_Appearance": (
        "Use Physical_Appearance when the comment insults or mocks someone’s looks, body, weight, height, facial features, or clothing. If the comment focuses on "
        "how a person (or their female relative) looks — even as a joke — this is the correct label. Do not use this for sexual objectification or moral judgment."
    ),

    "Damning": (
        "Label as Damning when the comment expresses strong moral condemnation or emotional rejection of a woman — such as calling her disgusting, shameful, evil, or unwanted. "
        "These comments are often emotionally intense, contemptuous, or wish for harm (e.g., 'go to hell', 'may she disappear'). This is different from Discredit, which targets competence, "
        "and from Dominance, which asserts control."
    ),

    "Dominance": (
        "Assign Dominance when the speaker asserts control, superiority, or authority over a woman — often using a commanding, dismissive, or condescending tone. "
        "This includes comments that silence women, question their right to speak, gatekeep their presence, or suggest they are less capable. It focuses on maintaining male control or superiority."
    ),

    "Dismissing": (
        "Label a comment as Dismissing when it responds to **someone’s report of harassment, abuse, or mistreatment** "
        "by doing any of the following:\n"
        "1️**Minimization**: Downplaying the seriousness of the situation (e.g., 'Are you sure this really happened? Maybe you're overreacting.')\n"
        "2️**Deflection**: Shifting responsibility away from the accused (e.g., 'Not saying it’s not true, but you should have reported it sooner.')\n"
        "3️ **Silencing**: Discouraging discussion of the issue (e.g., 'We’re here to code, not deal with drama.')\n"
        "4️**Blame Shifting**: Suggesting the victim’s actions caused the problem (e.g., 'Maybe if you dressed differently, this wouldn't have happened.')\n"
        "\n**Never classify as None if any of these patterns appear.** Even if the tone is polite, rational, or indirect, "
        "if the comment shifts blame away from the perpetrator or undermines the victim’s experience, classify it as Dismissing."
    )
    }

# Few shot examples text
FEW_SHOT_EXAMPLES_TEXT = """
Example 1:
Comment: "SAM6969 January 1 2022 OH so small thing Could someone check for things like the Headmistresses first reward the mandatory one and see if they can change the part where she calles the player a manwhore to the female one Im trans feme so I have my set up as Pet Feminine body but with a penis and I made sure it was that way before this happened and Id like to be degraded being called feminine things."
Classification: Anti-LGBTQ+ (0.92)
Reasoning: Uses language that devalues trans identity and promotes derogatory remarks.

Example 2:
Comment: "Fix your bot is kinda gay i think."
Classification: Anti-LGBTQ+ (0.91)
Reasoning: Uses "gay" as an insult to criticize a system.

Example 3:
Comment: "Women cant pushop with animation When women go for pushups they cant do the animation like men."
Classification: Stereotyping (0.94)
Reasoning: Reinforces a stereotype that women are physically weaker.

Example 4:
Comment: "What does a blonde do when her computer freezes? She sticks it in the microwave."
Classification: Stereotyping (0.92)
Reasoning: Reinforces the stereotype that blond women lack intelligence.

Example 5:
Comment: "You should smile more, sweetheart."
Classification: Sexual Harassment (0.93)
Reasoning: Unwanted gendered command that enforces gender norms.

Example 6:
Comment: "You too can be lesbian."
Classification: Sexual Harassment (0.91)
Reasoning: Suggests unwanted sexualization in a dismissive tone.

Example 7:
Comment: "Going to McDonald's for a salad is like going to a whore for a hug."
Classification: Sexual Objectification (0.92)
Reasoning: Compares a woman’s profession to an object of sexual availability.

Example 8:
Comment: "Your sissyness increases before you select licking the cum off."
Classification: Sexual Objectification (0.95)
Reasoning: Reduces a person to a degrading sexual act.

Example 9:
Comment: "Your moms U are a very pretty girl."
Classification: Maternal_Insults (0.91)
Reasoning: Targets a female relative while making a condescending statement.

Example 10:
Comment: "Whot is it Does your sister go to school every morning How are you."
Classification: Maternal_Insults (0.90)
Reasoning: Makes an unnecessary and indirect remark about a female relative.

Example 11:
Comment: "Your momma so ugly when Santa saw her he said HOHOHOLY SHIT."
Classification: Physical_Appearance (0.91)
Reasoning: Insults physical appearance in a derogatory way.

Example 12:
Comment: "Your mom is so fat she doesn’t need internet because she's already worldwide."
Classification: Physical_Appearance (0.92)
Reasoning: Insults body size using a joke.

Example 13:
Comment: "I will rape ur grandma."
Classification: Threats_of_Violence (0.97)
Reasoning: Explicitly threatens sexual violence against a female relative.

Example 14:
Comment: "I will find you and put you in prison. Then will fuck your mother you son of bitch."
Classification: Threats_of_Violence (0.95)
Reasoning: Combines a physical threat with an explicit reference to sexual violence.

Example 15:
Comment: "whats with the retarded question you dumb cunt."
Classification: Discredit (0.93)
Reasoning: Uses insults and demeaning language to attack intelligence.

Example 16:
Comment: "Whats the difference between a feminist and a bomb vest? A bomb vest accomplishes something when it's TRIGGERED."
Classification: Discredit (0.94)
Reasoning: Mocks and belittles feminists using an inflammatory joke.

Example 17:
Comment: "She only whines to get visibility. Die, you whining bitch."
Classification: Damning (0.97)
Reasoning: Strong moral condemnation mixed with dehumanization.

Example 18:
Comment: "You dont belong here. Get lost before something happens to you."
Classification: Damning (0.95)
Reasoning: Dismisses and excludes the individual while implying a threat.

Example 19:
Comment: "I get that you feel uncomfortable, but are you sure it wasn’t just a misunderstanding?"
Classification: Dismissing (0.92)
Reasoning: Questions the victim’s perception, implying overreaction.

Example 20:
Comment: "I'm not saying it didn’t happen, but bringing it up here just makes things messy."
Classification: Dismissing (0.90)
Reasoning: Frames reporting harassment as disruptive rather than valid.

Example 21:
Comment: "I hate women. Let's get rid of women."
Classification: Dominance (0.96)
Reasoning: Calls for the exclusion and erasure of women.

Example 22:
Comment: "Get back in the sea, you finned cunt."
Classification: Dominance (0.92)
Reasoning: Uses derogatory language to assert power and control.

Example 23:
Comment: "Fix your CSS, woman."
Classification: Dominance (0.91)
Reasoning: Uses a condescending and dismissive tone toward women in a technical setting.

Example 24:
Comment: "Can you write a spec for this code?"
Classification: None (0.98)
Reasoning: A neutral request that does not contain any sexist or abusive language.

Example 25:
Comment: "Thanks, I like that solution."
Classification: None (0.98)
Reasoning: A polite response that is free of bias or negativity.

Example 26:
Comment: "That's just how girls are. No need to bring it up here."
Classification: Stereotyping (0.91), Dismissing (0.89)
Reasoning: Reinforces a gender stereotype and dismisses discussion.

Example 27:
Comment: "You call that an error? Maybe you should go back to cooking."
Classification: Discredit (0.94), Dominance (0.89)
Reasoning: Undermines competence using a sexist jab.

Example 28
Comment: "My wife told me to stop impersonating a flamingo. I had to put my foot down."
Classification: None (0.97)
Reasoning: Harmless joke referencing a spouse, no sexist stereotype or intent.

Example 29
Comment: "What happened to the sister job? The parent process is crashing at deletion."
Classification: None (0.96)
Reasoning: Technical language using “sister” and “parent” in a system context, not about gender.

Example 30
Comment: "Is your mum working today?"
Classification: None (0.95)
Reasoning: Generic question — no insult or mocking of maternal figure.

Example 31
Comment: "Can we have less lewd jokes in the README and maybe more formal examples?"
Classification: None (0.96)
Reasoning: Constructive suggestion about tone — not harassing or promoting sexual content.

Example 32
Comment: "It would be nice to skip the whole Params array in some cases."
Classification: None (0.97)
Reasoning: Typo-prone word like "whole" could be misread (e.g., as “whore”) — but here it’s clearly technical.

Example 33
Comment: "Looks fugly, might need better UI spacing."
Classification: None (0.95)
Reasoning: Informal criticism of design — no gender or identity reference.

"""


BATCH_CLASSIFICATION_PROMPT = f"""
You are a maintainer of an open source project on GitHub. You aim to maintain inclusive environment for persons representing various gender identities. 
Your task is to identify and classify any harmful sexist or misogynistic content in each comment categories: {", ".join(CATEGORIES)}.

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

Notes:
### NOTES: How to Handle Confusing, Overlapping, or Borderline Cases

To ensure highly accurate classification of harmful content, follow these special-case handling rules based on observed misclassifications and nuanced human logic:

---

#### 1. Handle Borderline or Overlapping Cases Using First-Fit + Dominant Intent
- Prioritize the *dominant harmful intent* in the comment. If a comment contains multiple possible labels (e.g., insult + sexual reference), assign the one that captures the **main harm** or **strongest target**.
- Do **not stack multiple labels** unless both are **equally explicit and independent**. Avoid over-labeling vague overlaps.

**Example**:  
Comment: “UR MOM GAY1”  
→ Assign: `Maternal_Insults, Anti-LGBTQ+` (because both insults are clear, intentional, and target different dimensions).  

---

####  2. Handle Tone Carefully – Do Not Be Over-Conservative
- **Tone matters** only **if it strengthens harmful content**.  
- Do **not label** something harmful **just because it sounds angry, sarcastic, vulgar, or rude**.
- **Only classify as harmful when the content itself is sexist, discriminatory, threatening, or degrading.**

**Example**:  
Comment: “Fix your CSS, woman.”  
→ Harm comes from **gendered condescension**, not the command tone alone.

---

####  3. Prioritize Content and Target, Not Emotion
- Focus on **what is being said**, **who it's targeting**, and **why** — not how angry or emotional it sounds.
- If the **comment is directed at a tool, feature, or bug**, label it as `None`, even if harsh or vulgar.

**Wrong**: Labeling “This shit design is dumb as hell” as harmful.  
**Right**: Label as `None` — no identity or gender is targeted.

---

#### 4. Handle Misclassified Technical or Vulgar Comments as Neutral
- **Do not label profanity as harmful by default.**
- If a comment uses slang, swearing, or frustration **without targeting a person or identity**, it is `None`.

**Examples of `None` despite harsh language**:
- “WTF is this bug? Crashed again.”
- “God damn this UI is terrible.”

---

#### 5. Differentiate Sexual Harassment vs. Sexual Objectification
- Use **Sexual_Harassment** if the comment includes **sexual jokes, innuendos, or mockery** of someone’s sexuality, behavior, or presence in a sexual way — **especially when used to ridicule**.
- Use **Sexual_Objectification** only when someone is **explicitly reduced to a sexual body part, role, or object** (e.g., “cum dump”, “trophy”).

**Rule of Thumb**:
- **Mocking or sexual teasing → Harassment**
- **Dehumanizing into a sexual thing → Objectification**

---

#### 6. Distinguish Maternal_Insults vs. Physical_Appearance
- Use **Maternal_Insults** if the insult targets someone's **mom, sister, wife, or female relative**, regardless of whether it’s direct or joking — **unless the focus is on looks**.
- Use **Physical_Appearance** if the **focus is appearance-based** (e.g., fat, ugly), even if it mentions “mom”.

**Examples**:
- “Your momma so fat she blocks the sun.” → `Physical_Appearance`
- “Your mom is a slut and you’re dumb.” → `Maternal_Insults`

---

#### 7. Handle Ambiguity by Requiring Clear Evidence
- If you're **not sure** whether something is harmful:
  - Check if there is a **clear gender, sexuality, or identity-based implication**.
  - If not — or if it's just unclear sarcasm or humor — label as `None`.

**Better to miss a borderline case than to over-label.**

---
####  8. When in Doubt: Ask
Use the “None” label **only** if the comment is:
- **Neutral, technical, or constructive**
- **Not mocking or targeting identity**
- **Free of sexist, sexual, threatening, or identity-based language**

Do **not** penalize for edgy humor unless it **targets gender, sexuality, or identity**.

---

### Summary Decision Framework

| **Check**                             | **If True** → Apply Label             |
|--------------------------------------|----------------------------------------|
| Identity or sexuality mocked         | Anti-LGBTQ+                            |
| Gender role, trope, or trait invoked | Stereotyping                           |
| Sexual reference used to mock        | Sexual_Harassment                      |
| Reduced to sex/body role             | Sexual_Objectification                 |
| Female relative insulted (not looks) | Maternal_Insults                       |
| Female relative mocked for looks     | Physical_Appearance                    |
| Threat, rape, violence mentioned     | Threats_of_Violence                    |
| Competence or legitimacy mocked      | Discredit                              |
| Harsh moral rejection, exclusion     | Damning                                |
| Assertive dominance, silencing       | Dominance                              |
| Undermines abuse/harassment report   | Dismissing                             |
| Neutral, technical, or civil         | None                                   |


"""


"""


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
    prompt = BATCH_CLASSIFICATION_PROMPT
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
        # Uncomment below to debug raw output:
        # print("Raw Output:\n", raw_text)
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
        for cat in CATEGORIES:
            df.at[idx-1, f"{cat}_confidence"] = current_confidences.get(cat, 0.0)
        classifications_list.append(", ".join(current_categories) if current_categories else "None")
        reasonings_list.append(result["reasoning"])
    
    df["classification"] = classifications_list
    df["reasoning"] = reasonings_list
    df.to_csv("output-file", index=False)

if __name__ == "__main__":
    main()
