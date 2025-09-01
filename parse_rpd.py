#%%
# parse
import pandas as pd
import src.utils as utils

#%%
parse_prompt = """Extract the name of the discipline/course and the names of the topics covered in this course, all in Russian. Only include academic topics, not administrative. Respond only with the names of topics, separated by comma. Put the name of the discipline/course first, before the first topic."""

#%%
#parse RPD
client = utils.get_api_client()
parsed_topics = utils.parse_document("https://www.hse.ru/ba/math/courses/920973298.html", parse_prompt, client)

topics = parsed_topics.split(", ")
discipline = topics[0]
topics = topics[1:]
print("Discipline:", discipline)
print("Topics:", topics)

#%%
# embed topics
text_to_embed = f"{discipline}, {', '.join(topics)}"
embedding = utils.embed_text(text_to_embed, client)

#%%
embeddings_by_id = utils.load_course_embeddings()
most_similar = utils.get_most_similar(embedding, embeddings_by_id, top_k=5)
# most_similar is list of (id, score) tuples
most_similar_ids = [item[0] for item in most_similar]

# build id -> score map and load courses
similarity_map = {item[0]: item[1] for item in most_similar}
courses_df = pd.read_csv('courses.csv')

# filter and add similarity column, then sort by similarity
similar_courses_df = courses_df[courses_df['project_id'].isin(most_similar_ids)].copy()
similar_courses_df['similarity'] = similar_courses_df['project_id'].map(similarity_map).astype(float)
similar_courses_df = similar_courses_df.sort_values('similarity', ascending=False).reset_index(drop=True)

similar_courses_df

#%%
for (i, row) in similar_courses_df.iterrows():
    course_name = row['project_name']
    course_topics = row['topics']
    print(f"Title: {course_name}, Topics: {course_topics}")
    decision, explanation = utils.determine_course_suitability(discipline, topics, course_name, course_topics, client)
    print(f"Suitable: {decision}\nExplanation: {explanation}\n")
# %%
