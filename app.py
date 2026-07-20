#============================
#imports
#============================
from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process
import re
import json
#===========================

#=====================
#Flask app
#==========================

app = Flask(__name__)
user_context = {
    "skills": "",
    "qualification": "",
    "experience": "",
    "recommended_careers": [],
    "recommendation_done": False
}
def ask_llm(prompt):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False,
        
        "options": {
            "temperature": 0.2,
            "num_predict": 180,
            "top_p": 0.9
        }
    }

    try:
        response = requests.post(url, json=payload,timeout=90)
        response.raise_for_status()

        data = response.json()
        return data["response"]

    except Exception as e:
        print("Ollama Error:", e)
        return None
@app.route("/")
def home():
    
    return render_template("index.html")
#=========================
# Load dataset

data = pd.read_csv("dataset.csv")
#========================
#skill mapping
#========================

skill_to_career = {}

for _, row in data.iterrows():

    career = row["recommend_career"]

    skills = [
        s.strip().lower()
        for s in str(row["skills"]).split(",")
    ]

    if career not in skill_to_career:
        skill_to_career[career] = set()

    skill_to_career[career].update(skills)
# ==========================
# Build skills dynamically from dataset
# ==========================

all_skills = []

for skills in data["skills"]:

    for skill in str(skills).split(","):

        skill = skill.strip().lower()

        if skill not in all_skills:
            all_skills.append(skill)

vectorizer = TfidfVectorizer( lowercase=True, ngram_range=(1,2))
skill_vectors = vectorizer.fit_transform(all_skills)
career_texts = data["skills"].fillna("").tolist()

career_vectorizer = TfidfVectorizer(lowercase=True)

career_vectors = career_vectorizer.fit_transform(career_texts)

def extract_skills(user_text):

    detected_skills = []

    threshold = 0.60
    # Expand common abbreviations
    user_text = user_text.lower()

    user_text = re.sub(r"\bml\b", "machine learning", user_text)
    user_text = re.sub(r"\bai\b", "machine learning", user_text)
    user_text = re.sub(r"\bdbms\b", "sql", user_text)
    user_text = re.sub(r"\bsql server\b", "sql", user_text)
    user_text = re.sub(r"\bdl\b", "deep learning", user_text)
    user_text = re.sub(r"\bjs\b", "javascript", user_text)
    # Split user input into individual skills
    user_skills = [
        
        skill.strip().lower()
        for skill in re.split(r",|\band\b|&", user_text)
        if skill.strip()
        
    ]


    for user_skill in user_skills:

        input_vector = vectorizer.transform([user_skill])

        similarities = cosine_similarity(
            input_vector,
            skill_vectors
        ).flatten()

        best_index = similarities.argmax()
        best_score = similarities[best_index]
        
        if best_score >= threshold:

           skill = all_skills[best_index]

        else:
         # Fuzzy matching for spelling mistakes
         match = process.extractOne(user_skill, all_skills)

         if (
             match and match[1] >= 80 and abs(len(user_skill) - len(match[0])) <=2
            ):
              skill = match[0]
         else:
          continue

        if skill not in detected_skills:

         detected_skills.append(skill)
    return (
        detected_skills
        
    )
@app.route("/get_recommendation", methods=["POST"])
def recommend():
    best_row = None
    max_score = 0
    career_scores = {}
    top_careers = []
    career_output = []
    user_data = request.json

    skills = user_data["skills"]
    original_skills=skills
    qualification = user_data["qualification"]
    experience = user_data["experience"]

    # ===== Testing Logs =====
    print("\n" + "="*45)
    print("          New User Request")
    print("="*45)
    print("User Skills      :", skills)
    print("Qualification    :", qualification)
    print("Experience       :", experience)
    print("="*45)

    detected_skills = extract_skills(skills)
    user_skill_count = len([
    s.strip()
    for s in re.split(r",|\band\b|&", original_skills)
    if s.strip()
    ])

    known_skill_count = len(detected_skills)

    unknown_skill_count = user_skill_count - known_skill_count
    total_skills = len(detected_skills) + unknown_skill_count

    if total_skills > 0:
        feature_accuracy = (len(detected_skills) / total_skills) * 100
    else:
        feature_accuracy = 0
    # Decide which system to use
    use_llm = unknown_skill_count >= known_skill_count
    
    
    skills = ",".join(detected_skills)

    # qualification = user_data["qualification"]

    # experience = user_data["experience"]
    
    # Normalize qualification
    qualification = qualification.strip().lower()

    qualification_map = {
    "mca": "master",
    "m.tech": "master",
    "mtech": "master",
    "bca": "bachelor",
    "b.tech": "bachelor",
    "btech": "bachelor",
    "bsc": "bachelor",
    "b.sc": "bachelor"
    }

    qualification = qualification_map.get(qualification, qualification)

    # Normalize experience
    experience = experience.strip().lower()

    experience_map = {
    "fresher": "entry",
    "0-1 years": "entry",
    "0-2 years": "entry",
    "1-3 years": "mid",
    "2-5 years": "mid",
    "5+ years": "senior"
    }

    experience = experience_map.get(experience, experience)


    if not use_llm:
      career_scores = {}
      career_counts = {}
      best_rows = {}
      max_score = 0
      top_careers = []
      best_row = None

      for index, row in data.iterrows():

        career_name = row["recommend_career"]

        dataset_skills = [
          s.strip().lower()
          for s in str(row["skills"]).split(",") 
        ]
        

        user_skills_list = [
          s.strip().lower()
          for s in skills.split(",")
          if s.strip()
        ]

        matched_skills = set(user_skills_list).intersection(dataset_skills)

        # Skip rows with no matching skills
        if len(matched_skills) == 0:
           continue

        skill_score = len(matched_skills) / len(dataset_skills) * 100
        qualification_score = 0
        if qualification in str(row["qualification"]).lower():
           qualification_score = 10

        experience_score = 0
        if experience in str(row["experience_level"]).lower():
           experience_score = 10

        score = (
           skill_score * 0.8
           + qualification_score
           + experience_score
        )

        

        if career_name not in career_scores:
          career_scores[career_name] = score
          career_counts[career_name] = 1
          best_rows[career_name] = row

        else:
          career_scores[career_name] += score
          career_counts[career_name] += 1

          # Keep highest row for roadmap
          if score > career_scores[career_name] / career_counts[career_name]:
             best_rows[career_name] = row
             
      for career in career_scores:
        career_scores[career] = (
        career_scores[career] /
        career_counts[career]
      )
      # Highest scoring career
      if career_scores:
        best_match = max(career_scores, key=career_scores.get)
        max_score = career_scores[best_match]
        best_row = best_rows[best_match]
      else:
        max_score = 0
        best_row = None
        

    learning_resources = {}
    roadmap = []
    missing_skills = []
    step = 1 
    

    if best_row is not None:

        dataset_skills = [
            s.strip()
            for s in str(best_row['skills']).lower().split(",")
        ]

        user_skills_list = [
            s.strip()
            for s in skills.lower().split(",")
        ]

        missing_skills = list(set(dataset_skills) - set(user_skills_list))
        display_missing_skills = []

        for skill in missing_skills:

            if skill.lower() == "r":
               display_missing_skills.append("R Programming")
            else:
               display_missing_skills.append(skill)

        missing_skills = display_missing_skills
    
        

        for skill in missing_skills:

          roadmap.append(f"Step {step} → Learn {skill}")

          step += 1

    roadmap.append(f"Step {step} → Build Projects")
    step += 1

    roadmap.append(f"Step {step} → Apply for Internships")
        

    for skill in missing_skills:
            
            search_skill = skill.replace(" ", "+")

            learning_resources[skill] = {

                "YouTube":
                f"https://www.youtube.com/results?search_query={search_skill}+tutorial",

                "Coursera":
                f"https://www.coursera.org/search?query={search_skill}",

                "GeeksforGeeks":
                f"https://www.google.com/search?q=site:geeksforgeeks.org+{search_skill}+tutorial"
            }
    sorted_careers = sorted(
        career_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_careers = sorted_careers[:3] 
    career_output = []

    for career, score in top_careers:

        percentage = min(int(score), 100)
        if percentage > 0:
              career_output.append({
                  "career": career,
                  "reason": f"{percentage}% Match"
              })  

    if  use_llm or max_score < 15:

       user_skills_list = [
         s.strip()
         for s in skills.lower().split(",")
        ]
       llm_confidence = 40

       if detected_skills:
          llm_confidence += min(len(detected_skills) * 10, 30)

       if qualification:
          llm_confidence += 15

       if experience:
          llm_confidence += 15

       prompt = f"""
        User skills:
       {original_skills}

       Recommend exactly 3 careers.

       Output format:

       Journalist - Writes and reports news.
       Content Writer - Creates engaging digital content.
       Public Relations Specialist - Builds strong public communication.

       Rules:
       - Return exactly 3 careers.
       - Each career must be on a single line.
       - Format:
         Career - Reason
       - Use exactly one "-" between career and reason.
       - Do NOT write the reason on a new line.
       - If necessary, infer a suitable third career.
       - No numbering.
       - No bullets.
       - No markdown.
       """
       try:

          ai_reply = ask_llm(prompt)

          career_list = []

          if ai_reply:

             for line in ai_reply.split("\n"):

                 line = line.strip()

                 if "-" not in line:
                    continue

                 parts = line.split("-", 1)

                 if len(parts) != 2:
                    continue

                 career_list.append({
                   "career": parts[0].strip(),
                   "reason": parts[1].strip()
                 })

          fallback_careers = career_list
          if len(career_list) == 3:
             llm_confidence += 10
             llm_confidence = min(llm_confidence, 100)
        
          skill_prompt = f"""
          User Skills:
          {original_skills}

          Recommended Careers:
          {",".join([c["career"] for c in career_list])}

          Suggest ONLY the top 3 technical or professional skills that the user should learn to become successful in these recommended careers.
          Rules:
          Return exactly 3 skills.
          Only skill names.
          Do NOT write "Learn".
          one skill per line.
          No numbering or explanations.
          """
          skill_reply = ask_llm(skill_prompt)

          llm_missing_skills = []

          # Create once
          user_skill_set = {
               s.strip().lower()
               for s in detected_skills
            }

          if skill_reply:
               for line in skill_reply.split("\n"):

                   skill = line.strip()
                   skill = skill.replace("Learn ", "")
                   skill = skill.replace("-", "")
                   skill = skill.strip()

                   if (
                       skill
                       and skill.lower() not in user_skill_set
                       and skill.lower() not in [s.lower() for s in llm_missing_skills]
                    ):
                    llm_missing_skills.append(skill)
          
          roadmap_prompt = f"""
          Recommended Careers:
          {",".join([c["career"] for c in career_list])}

          Skills to Learn:
          {",".join(llm_missing_skills)}

          Create a personalized learning roadmap.

          Rules:
          - Exactly 5 steps.
          - Maximum 8 words per step.
          - Start every line with:
          Step 1 →
          Step 2 →
          Step 3 →
          Step 4 →
          Step 5 →
          - No explanations.
          - No paragraphs.
          """

          roadmap_reply = ask_llm(roadmap_prompt)

          llm_roadmap = []

          if roadmap_reply:
             for line in roadmap_reply.split("\n"):
                 line = line.strip()
                 if line:
                  llm_roadmap.append(line)
          llm_learning_resources = {}

          for skill in llm_missing_skills:

              search_skill = skill.replace(" ", "+")

              llm_learning_resources[skill] = {

                 "YouTube":
                 f"https://www.youtube.com/results?search_query={search_skill}+tutorial",

                 "Coursera":
                 f"https://www.coursera.org/search?query={search_skill}",

                 "GeeksforGeeks":
                 f"https://www.google.com/search?q=site:geeksforgeeks.org+{search_skill}+tutorial"
              }
           

       except Exception as e:
           print("error:", e)

           fallback_careers = [
        "AI could not generate a recommendation. Please try again."
             ]
       user_context["skills"] = skills
       user_context["qualification"] = qualification
       user_context["experience"] = experience

       user_context["recommended_careers"] = fallback_careers

       user_context["recommendation_done"] = True 
       total_skills = len(detected_skills) + unknown_skill_count

       if total_skills > 0:
          feature_accuracy = (len(detected_skills) / total_skills) * 100
       else: 
          feature_accuracy = 0
       
       print("\n" + "="*45)
       print("         PathFinder Summary")
       print("="*45)
       print("Recommendation Engine :", "LLM")
       print("Detected Skills       :", len(detected_skills))
       print("Unknown Skills        :", unknown_skill_count)
       print("Feature Accuracy      :", f"{feature_accuracy:.2f}%")
       print("LLM Confidence        :", f"{llm_confidence}%")
       print("="*45)

       return jsonify({

           "career": fallback_careers,

           "detected_skills": original_skills.split(","),

           "missing_skills": llm_missing_skills,

           "learning_resources": llm_learning_resources,

           "roadmap": llm_roadmap

    })

    else:
        user_context["skills"] = skills
        user_context["qualification"] = qualification
        user_context["experience"] = experience

        user_context["recommended_careers"] = [
        career for career, score in top_careers
        ]

        user_context["recommendation_done"] = True
        if total_skills > 0:
           feature_accuracy = (len(detected_skills) / total_skills) * 100
        else:
           feature_accuracy = 0
        
        total_skills = len(detected_skills) + unknown_skill_count
        print("\n" + "="*45)
        print("         PathFinder Summary")
        print("="*45)
        print("Recommendation Engine :", "Dataset")
        print("Detected Skills       :", len(detected_skills))
        print("Unknown Skills        :", unknown_skill_count)
        print("Feature Accuracy      :", f"{feature_accuracy:.2f}%")
        print("Recommendation Score  :", f"{max_score:.2f}%")
        print("="*45)

        return jsonify({

        "career": career_output,

        "detected_skills": detected_skills,

        "missing_skills": missing_skills,

        "learning_resources": learning_resources,

        "roadmap": roadmap

     })
     
@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()
    user_message = data["message"]

    if not user_context["recommendation_done"]:
        return jsonify({
            "reply": "👋 Please enter your skills first so I can recommend a suitable career path."
        })

    prompt = f"""
    You are PathFinder, an AI Career Assistant.

    User Profile:
    Skills: {user_context["skills"]}
    Qualification: {user_context["qualification"]}
    Experience: {user_context["experience"]}
    Recommended Careers: {", ".join(c["career"] for c in user_context["recommended_careers"])}
    Rules:
    - Answer only career-related questions.
    - Base every answer on the user's profile and recommended careers.
    - Keep replies within 5-6 lines.
    - Be direct and concise.
    - Use simple English.
    - Recommend only relevant skills,certifications, or 3-5 companies when asked.
    - For salary-related questions, always provide the salary in INR (₹) and LPA. Do not use USD unless the user explicitly asks for salaries in another country.

    User Question:
    {user_message}
    """

    answer = ask_llm(prompt)

    return jsonify({
        "reply": answer
    })
if __name__ == "__main__":
    app.run(debug=False)