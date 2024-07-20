import pandas as pd   
from ragas import evaluate 
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from datasets import Dataset


embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
llm = OllamaLLM(model='mistral') 

df = pd.read_csv('./evaluations/results.csv')
df['contexts'] = df.apply(lambda row: eval(row['text']) + eval(row['image_text']), axis=1)

data = {
    'question': df['question'].values,
    'answer': df['answer'].values,
    'contexts': df['contexts'].values,
    'ground_truth': df['ground_truth'].values,
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, 
             answer_relevancy, 
             context_precision, 
             context_recall, 
             context_entity_recall, 
             answer_similarity, 
             answer_correctness
             ],
    llm=llm,
    embeddings=embed_model
)

eval_df = result.to_pandas()
eval_df.to_excel('./evaluations/evaluation_results.xlsx')

print(eval_df[['faithfulness','answer_relevancy', 'context_precision', 'context_recall',
       'context_entity_recall', 'answer_similarity', 'answer_correctness']].mean(axis=0))





