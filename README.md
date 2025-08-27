# Urait course matching

## Pipeline description

to answer the question "what courses does Urait have for a given university specialization?" we:

- parse the disciplines from the university URL
- generate topics for each discipline using Gemini
- find 5 most similar courses by cosine similarity using Gemini embeddings
- ask Gemini if any of these courses can be used to teach the discipline

The reason for generating topics is for better matching, since the discipline name alone might be not enough for good embeddings. The reason for asking Gemini about the top 5 courses is that cosine similarity might not be perfect, and we want to give Gemini a chance to choose the best course itself.

## How to run

0. Run `pip install -r requirements.txt`
1. download `project_subjects.csv` from [drive](https://drive.google.com/drive/folders/16_rbQxV5SVpZemgS0NN0-Odo4ORpZXfv).
2. make a `.env` file with `GOOGLE_API_KEY=your_key`.
3. run `embed_courses.py` to create the course embeddings.
4. run `parse_disciplines.py` to parse disciplines from the university URL and decide whether there is a course that can be used to teach it.
