from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
from keybert import KeyBERT, KeyLLM
from sentence_transformers import SentenceTransformer
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import json
import textwrap
#from llmware import store_verified_response
import key_param

client = MongoClient(key_param.MONGO_URI)
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]


# Define the text embedding model
 
embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)

# Initialize the Vector Store

vectorStore = MongoDBAtlasVectorSearch( collection, embeddings, index_name="vector_index")

def extract_keywords(query,input_year):
    # Initialize KeyBERT
    kw_model = KeyBERT()
    document_content=extract_summary(query,input_year)
    # Extract keywords
    keywords = kw_model.extract_keywords(document_content, stop_words=None)

    return keywords

# def extract_keywords(document_content):
#     # Initialize KeyBERT
#     kw_model = KeyBERT()

#     # Extract keywords
#     keywords = kw_model.extract_keywords(document_content, stop_words=None)

#     return keywords

def extract_summary(query,input_year):
    llm = OpenAI(openai_api_key=key_param.openai_api_key, temperature=0)


    # Get VectorStoreRetriever: Specifically, Retriever for MongoDB VectorStore.
    # Implements _get_relevant_documents which retrieves documents relevant to a query.
    retriever = vectorStore.as_retriever(
        search_kwargs={'k': 2,'filter': {'metadata.date': {'$gte': int(input_year)}}}
    )

    # Load "stuff" documents chain. Stuff documents chain takes a list of documents,
    # inserts them all into a prompt and passes that prompt to an LLM.

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

    # Execute the chain

    retriever_output = qa.run(query)
    return retriever_output


def extract_history(query,input_year):
    
    docs = vectorStore.similarity_search_with_score(query, k=10)
    docs = [doc for doc in docs if doc[0].metadata.get('date') >= int(input_year)]
    print(docs[0])
    print(docs[0][0], docs[0][1]) # Document, Score

    as_output = docs[0][0].page_content

    top_score = docs[0][1]
    source = docs[0][0].metadata['source']

    sources = "\n".join([doc[0].metadata['source'] + f" (score: {doc[1]})" for doc in docs if doc[1] >= top_score-0.02])
    return sources


def opinion_scale(text_input,opinion_scale):
    pass




def query_data(query,input_year):
    # Convert question to vector using OpenAI embeddings
    # Perform Atlas Vector Search using Langchain's vectorStore
    # similarity_search returns MongoDB documents most similar to the query    
    print(input_year)
    
    #gia ta xronia thelei na mpoun san metadata
    #filtered_documents = [doc for doc in vectorStore if doc.metadata.get('year') >= input_year]
    #docs = vectorStore.similarity_search_with_score(query, k=10,docs=filtered_documents)
    
    docs = vectorStore.similarity_search_with_score(query, k=10)
    docs = [doc for doc in docs if doc[0].metadata.get('date') >= int(input_year)]
    print(docs[0])
    print(docs[0][0], docs[0][1]) # Document, Score

    as_output = docs[0][0].page_content

    top_score = docs[0][1]
    source = docs[0][0].metadata['source']

    source_list = [doc[0].metadata['source'] for doc in docs if doc[1] >= top_score-0.03]
    source_dicts = [{'link': doc[0].metadata['source'], 'score': doc[1]} for doc in docs if doc[1] >= top_score-0.03]
    sources = "\n".join([doc[0].metadata['source'] + f" (score: {doc[1]})" for doc in docs if doc[1] >= top_score-0.03])

    # Leveraging Atlas Vector Search paired with Langchain's QARetriever

    # Define the LLM that we want to use -- note that this is the Language Generation Model and NOT an Embedding Model
    # If it's not specified (for example like in the code below),
    # then the default OpenAI model used in LangChain is OpenAI GPT-3.5-turbo, as of August 30, 2023
    
    llm = OpenAI(openai_api_key=key_param.openai_api_key, temperature=0)


    # Get VectorStoreRetriever: Specifically, Retriever for MongoDB VectorStore.
    # Implements _get_relevant_documents which retrieves documents relevant to a query.
    retriever = vectorStore.as_retriever(
        search_kwargs={'k': 2,'filter': {'metadata.date': {'$gte': int(input_year)}}}
    )

    # Load "stuff" documents chain. Stuff documents chain takes a list of documents,
    # inserts them all into a prompt and passes that prompt to an LLM.

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

    # Execute the chain

    retriever_output = qa.run(query)

    keywords = extract_keywords(retriever_output)
    print("Extracted Keywords:", keywords)
    # Store the verified response and citations
    #store_verified_response(retriever_output, evidence_sources)

    download_as_json(query, retriever_output, source_dicts, keywords)
    print(source_list, keywords)
    download_as_pdf(query, retriever_output, source_list, keywords)

    # Return Atlas Vector Search output, and output generated using RAG Architecture
    return as_output, retriever_output, sources, keywords

# Define functions for handling download actions
def download_as_pdf(query, summary_output, sources, keywords):
    # Create a new PDF file
    print("----------------------------")
    print(sources)
    print(keywords)
    print(', '.join([kw[0] for kw in keywords]))
    filename = "output.pdf"
    c = canvas.Canvas(filename, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 700, "Query:")


    # Set font properties for titles
    c.setFont("Helvetica", 12)
    c.drawString(100, 680, query)

    # Add titles
    c.drawString(100, 650, "Summary Output:")
    
    

    # Add content (replace with your actual data)
    c.setFont("Helvetica", 12)

    max_line_length = 75
    y = 630
    summary_output_wrapped = textwrap.wrap(summary_output, width=max_line_length)
    print(summary_output_wrapped)
    for line in summary_output_wrapped:
        c.drawString(100, y, line)
        y -= 20

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, y-10, "Sources:")
    c.setFont("Helvetica", 12)
    y -=30
    for source in sources:
        c.drawString(100, y, source)
        y -= 20

    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, y-10, "Keywords:")
    c.setFont("Helvetica", 12)
    c.drawString(100, y-30, ', '.join([kw[0] for kw in keywords]) + ".")

    # Save the PDF
    c.save()

    print(f"PDF saved to {filename}")

def download_as_json(query, retriever_output, sources, keywords):
    data = {
    "query": query,
    "summary_output": retriever_output,
    "sources": sources,
    "keywords": keywords
    }
    filename = "output.json"
    with open(filename, "w") as json_file:
        json.dump(data, json_file, indent=4)
    return

# Create a web interface for the app, using Gradio

# with gr.Blocks(theme=Base(), title="Summarizing + Question Answering App using Vector Search + RAG") as demo:
#     gr.Markdown(
#         """
#         # Question Answering App using Atlas Vector Search + RAG Architecture
#         """)
#     year_select = gr.Dropdown(choices=[str(year) for year in range(2015, 2026)], label="Choose a Year")
#     textbox = gr.Textbox(label="Enter your Question:")
#     with gr.Row():
#         button = gr.Button("Submit", variant="primary")
#     with gr.Column():
#         output1 = gr.Textbox(lines=1, max_lines=10, label="Output with just Atlas Vector Search (returns text field as is):")
#         output2 = gr.Textbox(lines=1, max_lines=10, label="Output generated by chaining Atlas Vector Search to Langchain's RetrieverQA + OpenAI LLM:")
#         sources = gr.Textbox(lines=1, max_lines=10, label="Source(s)")
#         keywords = gr.Textbox(lines=1, max_lines=10, label="Keywords")

# # Call query_data function upon clicking the Submit button

#     button.click(query_data, inputs=[textbox, year_select], outputs=[output1, output2, sources, keywords])
# demo.launch()






with gr.Blocks() as app:
    gr.Markdown("### Knowledge Management System")
    with gr.Row():
        text_input = gr.Textbox(label="Enter text", lines=4, placeholder="Type here...")
        year_select = gr.Dropdown(choices=[str(year) for year in range(2015, 2026)], label="Choose a Year")
        opinion_input = gr.Slider(minimum=1, maximum=10, label="Rate the quality (1-10)")

    with gr.Row():
        summary_button = gr.Button("Extract Summary")
        keywords_button = gr.Button("Extract Keywords")
        history_button = gr.Button("Extract History")
        opinion_button = gr.Button("Submit Opinion")

    summary_output = gr.Textbox(label="Summary Output", lines=4, placeholder="Summary will appear here...")
    keywords_output = gr.Textbox(label="Keywords Output", lines=2, placeholder="Keywords will appear here...")
    history_output = gr.Textbox(label="History Output", lines=4, placeholder="Historical events will appear here...")
   
    summary_button.click(extract_summary, inputs=[text_input, year_select], outputs=summary_output)
    keywords_button.click(extract_keywords, inputs=[text_input, year_select], outputs=keywords_output)
    history_button.click(extract_history, inputs=[text_input, year_select], outputs=history_output)
    opinion_button.click(opinion_scale, inputs=[text_input, opinion_input])
     with gr.Row():
        u = gr.UploadButton("Upload a file", file_count="single", visible=False)
        download_pdf_button = gr.DownloadButton("Download as PDF", variant="primary", value="output.pdf")
        download_json_button = gr.DownloadButton("Download as JSON", variant="primary", value="output.json")
app.launch()





