from tqdm import tqdm 
import pandas as pd
import requests

def hit_api_and_generate_results(input_excel_path, output_excel_path):
    # Read the input Excel file
    df = pd.read_csv(input_excel_path)
    
    # Initialize lists to store results
    answers = []
    texts = []
    image_paths = []
    image_texts = []
    contexts = []
    
    # Iterate over each row in the DataFrame
    for index, row in tqdm(df.iterrows()):
        question = row['question']
        
        # Prepare the API request payload
        payload = {
            "question": question
        }
        
        # Make the API request
        response = requests.post(
            url='http://127.0.0.1:5000/answer',
            headers={'Content-Type': 'application/json'},
            json=payload
        )
        
        # Parse the response
        data = response.json()
        answer = data.get('answer')
        metadata = data.get('metadata', {})
        text_entries = metadata.get('text', [])
        image_entries = metadata.get('images', [])
        
        # Extract text and image details
        extracted_texts = [entry['text'] for entry in text_entries]
        extracted_image_paths = [entry['image_path'] for entry in image_entries]
        extracted_image_texts = [entry['text'] for entry in image_entries]
        
        # Combine texts and image texts into context
        context = '\n\n'.join(extracted_texts + extracted_image_texts)
        
        # Append results to lists
        answers.append(answer)
        texts.append(extracted_texts)
        image_paths.append(extracted_image_paths)
        image_texts.append(extracted_image_texts)
        contexts.append(context)
    
    # Add new columns to the DataFrame
    df['answer'] = answers
    df['text'] = texts
    df['image_path'] = image_paths
    df['image_text'] = image_texts
    df['context'] = contexts
    
    # Write the result to a new Excel file
    df.to_csv(output_excel_path, index=False)

# Example usage
input_csv_path = './evaluations/test_cases.csv'
output_csv_path = './evaluations/results.csv'
hit_api_and_generate_results(input_csv_path, output_csv_path)