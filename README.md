# Urait course matching

## Поиск новинок
- Какие дисциплины преподаются? (50% done)
- Какие из них не закрыты курсами Юрайт? (50% done)
- Как закрыть эти дыры? (0% done)

## Pipeline description

To answer the question "what courses does Urait have for a given university specialization?" we:

- search for the study plans (учебные планы) for the specialization, via google search
- parse the disciplines from the study plan page, using Gemini 2.5 Flash
- search for work programs (рабочие программы дисциплин) for each discipline
- parse the topics from the work program page, using Gemini 2.5 Flash
- embed the topics using Gemini embeddings
- embed the Urait courses using Gemini embeddings
- match the topics to the courses using cosine similarity
- for top 5 course matches, use Gemini 2.5 Flash to decide whether the course can be used to teach the discipline

## Plans
- match the study plans to the universities they came from, using the urls
- get ~10-100 study plans per specialization, to approximate the popularity of the disciplines
- use the popularity to sort the holes -- disciplines without matching courses

## How to run

1. run `pip install -r requirements.txt`
2. download `project_subjects.csv` from [drive](https://drive.google.com/drive/folders/16_rbQxV5SVpZemgS0NN0-Odo4ORpZXfv).
3. make a `.env` file with `GOOGLE_API_KEY=your_key` and `SERPER_API_KEY=your_key`.