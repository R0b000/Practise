from sentence_transformers import SentenceTransformer, util

# Load pre-trained Sentence-BERT model
model = SentenceTransformer('./paraphrase-MiniLM-L6-v2')

# Example essay and prompt
prompt = """"Prevention is better than cure."
Out of a country's health budget, a large proportion should be diverted from treatment to spending on health education and preventative measures.
To what extent do you agree or disagree with this statement?"""
essay = """
Modern medicine has evolved along two lines: prevention and cure. In this world, many people are dying from various types of health-related problems due to the lack of appropriate health education and preventive actions. That is why, according to my perception, government should spend a huge amount of money from health budget for preventive measures.

To begin with, it is evident that, in most of the country's health education is hardly imparted among the people. When people are made aware about the health-related issues it naturally implies prevention of umpteen diseases which can be avoided with basic alterations in lifestyles. For example, in most of developing countries, people are suffering from so many communicable disease because they do not know about disadvantages of having unhygienic food and water. Moreover, If the government can provide the information about basic preventive steps which can be inculcated at basic grassroot level then the chances of spread of various diseases are alleviated and hence this would improve the overall health of people. Also, another major reason is creating awareness would definitely consume less funds of the government and would have more positive impacts.

On the other hand, it should also be remembered that not all examples of modern disease are preventable or predictable, and it is critical to maintain research into cures for all diseases. Hence certain amount of health budget should be diverted in curing diseases as well. For example, life taking diseases such as cancer cannot be prevented with basic measures and hence there should be funds allocated to the cure and research of such gruesome diseases.

To conclude, it can be said that, the government should spend sufficient amount of money on treatment too, but high proportion of amount should be spent on health education and preventive measures so that the population of the country can live in a healthy environment with less price.
"""

# Compute embeddings for prompt and essay sentences
prompt_embedding = model.encode(prompt, convert_to_tensor=True)
essay_sentences = essay.split('. ')
essay_embeddings = model.encode(essay_sentences, convert_to_tensor=True)

# Compute similarity scores
similarities = util.pytorch_cos_sim(prompt_embedding, essay_embeddings).numpy()

# Evaluate relevance based on similarity scores
relevance_scores = [similarity[0] for similarity in similarities]
average_relevance_score = sum(relevance_scores) / len(relevance_scores)

print(f"Relevance Scores: {relevance_scores}")
print(f"Average Relevance Score: {average_relevance_score}")