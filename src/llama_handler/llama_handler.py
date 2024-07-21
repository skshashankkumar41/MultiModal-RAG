import logging
import qdrant_client
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.clip import ClipEmbedding 
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.utils.utils import extract_text_from_image_v3
from src.prompts.prompts import qa_prompt, qa_prompt_mm
from llama_index.llms.ollama import Ollama
from llama_index.multi_modal_llms.ollama import OllamaMultiModal
from llama_index.core.schema import ImageDocument
import os
import shutil

logger = logging.getLogger(__name__)

class LlamaHandler():
    """
    A class to handle text and image embeddings, storage, and retrieval using Llama and Qdrant.

    Attributes:
        embed_model (HuggingFaceEmbedding): Embedding model for text.
        image_embed_model (ClipEmbedding): Embedding model for images.
        llm (Ollama): Language model for processing queries.
        mm_llm (OllamaMultiModal): Multi-modal language model for processing queries.
    """
    def __init__(self, text_embed_model='sentence-transformers/all-mpnet-base-v2', image_embed_model='ViT-L/14', llm='mistral', mm_llm=None):
        print(f"Loading Models...")
        self.embed_model = HuggingFaceEmbedding(text_embed_model)
        self.image_embed_model = ClipEmbedding(model_name=image_embed_model,embed_batch_size=16)
        self.llm = Ollama(model=llm, request_timeout=500.0) if llm else None
        self.mm_llm = OllamaMultiModal(model=mm_llm, request_timeout=600.0) if mm_llm else None
        
    def _create_storage_context(self, text_store, image_store):
        """
        Create a storage context from the given text and image stores.

        Args:
            text_store (QdrantVectorStore): Vector store for text.
            image_store (QdrantVectorStore): Vector store for images.

        Returns:
            StorageContext: The storage context combining the text and image stores.
        """
        print(f"Creating storage context...")
        storage_context = StorageContext.from_defaults(vector_store=text_store, image_store=image_store)
        return storage_context 
    
    def _create_vector_stores(self, client, collection_name):   
        """
        Create a Qdrant vector store with the given client and collection name.

        Args:
            client (QdrantClient): The Qdrant client.
            collection_name (str): The name of the collection in Qdrant.

        Returns:
            QdrantVectorStore: The created vector store.
        """
        print(f"Creating vector store...")
        store = QdrantVectorStore(
                    client=client, collection_name=collection_name
                    )
        return store
    
    def _return_text_documents(self, pdf_path='./data/pdf', chunk_size=200, chunk_overlap=25):
        """
        Load and process text documents from a directory.

        Args:
            pdf_path (str): Path to the directory containing PDF files.
            chunk_size (int): Size of text chunks.
            chunk_overlap (int): Overlap between text chunks.

        Returns:
            list: List of processed text nodes.
        """
        print(f"Extracting Text from the pdf...")
        text_documents = SimpleDirectoryReader(pdf_path).load_data()
        node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(text_documents)
        return nodes
    
    def _return_img_documents(self, img_path='./data/table_images'):
        """
        Load and process image documents from a directory.

        Args:
            img_path (str): Path to the directory containing image files.

        Returns:
            list: List of processed image documents with extracted text metadata.
        """
        print(f"Extracting Images tables...")
        img_documents = SimpleDirectoryReader(img_path).load_data()
        ## TODO: Utilize Vision to Text LLM for image to text summary
        print("Extracting text from image using OCR...")
        for doc in img_documents:
            doc.metadata['text'] = extract_text_from_image_v3(doc.metadata.get('file_path'))
        return img_documents
    
    def pipeline_v1_persist(self):
        """
        Create and persist a multi-modal index for text and images with different embedding(version 1).

        Returns:
            None
        """
        if os.path.exists('./data/index'):
            shutil.rmtree('./data/index')
        client = qdrant_client.QdrantClient(path="./data/index")
        text_store = self._create_vector_stores(client=client, collection_name='text_collection_v1')
        image_store = self._create_vector_stores(client=client, collection_name='image_collection_v1')
        
        storage_context = self._create_storage_context(text_store, image_store)
        
        text_documents = self._return_text_documents()
        img_documents = self._return_img_documents()
        
        _ = MultiModalVectorStoreIndex(text_documents+img_documents, storage_context=storage_context, embed_model=self.embed_model,image_embed_model=self.image_embed_model)
        
        client.close()
        
        return None 
   
    def pipeline_v2_persist(self):
        """
        Create and persist a multi-modal index for text and images with same embedding(version 2).

        Returns:
            None
        """
        client = qdrant_client.QdrantClient(path="./data/index")
        text_store = self._create_vector_stores(client=client, collection_name='text_collection_v2')
        image_store = self._create_vector_stores(client=client, collection_name='image_collection_v2')
        
        storage_context = self._create_storage_context(text_store, image_store)
        
        text_documents = self._return_text_documents(chunk_size=50, chunk_overlap=10)
        img_documents = self._return_img_documents()
        
        _ = MultiModalVectorStoreIndex(text_documents+img_documents, storage_context=storage_context, embed_model=self.image_embed_model,image_embed_model=self.image_embed_model)
        
        client.close()
        
        return None 
    
    def pipeline_v3_persist(self):
        '''
        Seperate indexes for text and image and will utilize seperate index for query as well
        '''
        client = qdrant_client.QdrantClient(path="./data/index")
        text_store = self._create_vector_stores(client=client, collection_name='text_collection_v3')
        image_store = self._create_vector_stores(client=client, collection_name='image_collection_v3')
        
        storage_context = self._create_storage_context(text_store, image_store)
        
        text_documents = self._return_text_documents()
        img_documents = self._return_img_documents()

        
        _ = MultiModalVectorStoreIndex(text_documents, storage_context=storage_context, embed_model=self.embed_model,image_embed_model=self.image_embed_model)
        _ = MultiModalVectorStoreIndex(img_documents, storage_context=storage_context, embed_model=self.embed_model, image_embed_model=self.image_embed_model)
        
        client.close()
        
        return None 
    
    def query_engine_v1(self, query, similarity_top_k=2, image_similarity_top_k=2):
        """
        Query the multi-modal index (version 1) and retrieve relevant text and image documents.

        Args:
            query (str): The query string.

        Returns:
            tuple: Retrieved texts and images.
        """
        client = qdrant_client.QdrantClient(path="./data/index")
        
        text_store = self._create_vector_stores(client=client, collection_name='text_collection_v1')
        image_store = self._create_vector_stores(client=client, collection_name='image_collection_v1')
                        
        index = MultiModalVectorStoreIndex.from_vector_store(
            vector_store=text_store,
            embed_model=self.embed_model,
            image_vector_store=image_store,
            image_embed_model=self.image_embed_model,
        )
        
        retriever = index.as_retriever(similarity_top_k=similarity_top_k, image_similarity_top_k=image_similarity_top_k)
        retrieval_results = retriever.retrieve(query)
        
        
        retrieved_texts = []
        retrieved_images = []

        for doc in retrieval_results:
            if 'pdf' in doc.node.metadata.get('file_type'):
                retrieved_texts.append({
                    'text':doc.node.text,
                    'file_path': doc.node.metadata.get('file_path')
                })
            elif 'png' in doc.node.metadata.get('file_type'):
                retrieved_images.append({
                    'image_path': doc.node.metadata.get('file_path'),
                    'text': doc.node.metadata.get('text')
                })
        
        print(f"Text Retrieval Results: {retrieved_texts}")
        print(f"Image Retrieval Results: {retrieved_images}")
        
        client.close()
        
        return retrieved_texts, retrieved_images
    
    
    def query_engine_v2(self, query):
        """
        Query the multi-modal index (version 2) and retrieve relevant text and image documents.

        Args:
            query (str): The query string.

        Returns:
            tuple: Retrieved texts and images.
        """
        client = qdrant_client.QdrantClient(path="./data/index")
        
        text_store = self._create_vector_stores(client=client, collection_name='text_collection_v2')
        image_store = self._create_vector_stores(client=client, collection_name='image_collection_v2')
                        
        index = MultiModalVectorStoreIndex.from_vector_store(
            vector_store=text_store,
            embed_model=self.image_embed_model,
            image_vector_store=image_store,
            image_embed_model=self.image_embed_model,
        )
        
        retriever = index.as_retriever(similarity_top_k=2, image_similarity_top_k=2)
        retrieval_results = retriever.retrieve(query)
        
        
        retrieved_texts = []
        retrieved_images = []

        for doc in retrieval_results:
            if 'pdf' in doc.node.metadata.get('file_type'):
                retrieved_texts.append({
                    'text':doc.node.text,
                    'file_path': doc.node.metadata.get('file_path')
                })
            elif 'png' in doc.node.metadata.get('file_type'):
                retrieved_images.append({
                    'image_path': doc.node.metadata.get('file_path'),
                    'text': doc.node.metadata.get('text')
                })
                
        print("text_retrieval_results:::",retrieved_texts)       
        print("img_retrieval_results:::",retrieved_images) 
        
        client.close()
        
        return retrieved_texts, retrieved_images
    
    def query_engine_v3(self, query):
        client = qdrant_client.QdrantClient(path="./data/index")
        
        text_store = self._create_vector_stores(client=client, collection_name='text_collection_v3')
        image_store = self._create_vector_stores(client=client, collection_name='image_collection_v3')
                        
        text_index = MultiModalVectorStoreIndex.from_vector_store(
            vector_store=text_store,
            embed_model=self.embed_model,
        )
        
        img_index = MultiModalVectorStoreIndex.from_vector_store(
            vector_store=image_store,
            embed_model=self.image_embed_model,
        )
        
        text_retriever = text_index.as_retriever(similarity_top_k=3)
        text_retrieval_results = text_retriever.retrieve(query)
        
        img_retriever = img_index.as_retriever(image_similarity_top_k=3)
        img_retrieval_results = img_retriever.retrieve(query)
        
        retrieved_texts = []
        retrieved_images = []

        for doc in text_retrieval_results:
            if 'pdf' in doc.node.metadata.get('file_type'):
                retrieved_texts.append({
                    'text':doc.node.text,
                    'file_path': doc.node.metadata.get('file_path')
                })
        
        for doc in img_retrieval_results:
            if 'png' in doc.node.metadata.get('file_type'):
                retrieved_images.append({
                    'image_path': doc.node.metadata.get('file_path'),
                    'text': doc.node.metadata.get('text')
                })
        
        print(f"Text Retrieval Results: {retrieved_texts}")
        print(f"Image Retrieval Results: {retrieved_images}")
        
        client.close()
        
        return retrieved_texts, retrieved_images
    
    def _create_context(self, retrieved_texts, retrieved_images):
        """
        Create a context string from the retrieved text and image documents.

        Args:
            retrieved_texts (list): List of retrieved text documents.
            retrieved_images (list): List of retrieved image documents.

        Returns context
        """
        context = "Context: \n"
        context += "Textual contents: \n"
        for obj in retrieved_texts:
            context += obj['text'] + '\n\n'
        
        if retrieved_images:    
            context += "table contents: \n"
            for obj in retrieved_images:
                context += obj['text'] + '\n\n'
            
        return context
    
    def _create_image_documents(self, retrieved_images):
        """
        Create a list of ImageDocument objects from the retrieved images.

        Args:
            retrieved_images (list): List of dictionaries containing image metadata.

        Returns:
            list: List of ImageDocument objects.
        """
        image_documents = []
        for obj in retrieved_images:
            image_documents.append(ImageDocument(image_path=obj['image_path']))
            
        return image_documents
    
    def answer_engine(self, question):
        """
        Answer a question by querying the multi-modal index and using the language model.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The answer generated by the language model.
        """
        retrieved_texts, retrieved_images = self.query_engine_v1(question)
        context = self._create_context(retrieved_texts, retrieved_images)
        
        prompt = qa_prompt.format(context_str=context, query_str=question)
        
        print(f"Prompt: {prompt}")
        
        response = self.llm.complete(prompt)
        
        print(f"Response: {response.text}")
        
        output = {
            'answer': response.text,
            'metadata':{
                'text':retrieved_texts,
                'images': retrieved_images
            }
        }
        
        return output
    
    def multi_modal_answer_engine(self, question):
        """
        Answer a question by querying the multi-modal index and using the multi-modal language model.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The answer generated by the multi-modal language model.
        """
        retrieved_texts, retrieved_images = self.query_engine_v1(question)
        context = self._create_context(retrieved_texts, [])
        image_documents = self._create_image_documents(retrieved_images)
        prompt = qa_prompt_mm.format(context_str=context, query_str=question)
        
        print(f"Prompt: {prompt}")
        
        response = self.mm_llm.complete(prompt=prompt, image_documents=image_documents)
        
        print(f"Response: {response.text}")
        
        output = {
            'answer': response.text,
            'metadata':{
                'text':retrieved_texts,
                'images': retrieved_images
            }
        }
        
        return output
        
    
    def multi_modal_answer_engine_v3(self, question):
        """
        Answer a question by querying the multi-modal index and using the multi-modal language model.

        Args:
            question (str): The question to be answered.

        Returns:
            str: The answer generated by the multi-modal language model.
        """
        retrieved_texts, retrieved_images = self.query_engine_v3(question)
        context = self._create_context(retrieved_texts, [])
        image_documents = self._create_image_documents(retrieved_images)
        prompt = qa_prompt_mm.format(context_str=context, query_str=question)
        
        print(f"Prompt: {prompt}")
        
        response = self.mm_llm.complete(prompt=prompt, image_documents=image_documents)
        
        print(f"Response: {response.text}")
        
        output = {
            'answer': response.text,
            'metadata':{
                'text':retrieved_texts,
                'images': retrieved_images
            }
        }
        
        return output
        
        
        
