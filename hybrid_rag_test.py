import os
import uuid
from getpass import getpass
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

groq_api_key = getpass('Enter Groq API Key')
os.environ['GROQ_API_KEY'] = groq_api_key

groq_model = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def scrape_article(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle")

        html_content = page.content()
        browser.close()

        soup = BeautifulSoup(html_content, "html.parser")
        article_text = "\n".join([p.get_text() for p in soup.find_all(["p", "h1", "h2", "h3"])])
        return article_text

def generate_chunk_context(document_chunk):
    chunk_process_prompt = """
    You are an AI assistant optimizing SEO content using Retrieval-Augmented Generation (RAG).

    Given the following content chunk:
    <chunk>
    {chunk}
    </chunk>

    Provide a concise summary that captures its essence for better search retrieval.

    Response:
    """
    prompt_template = ChatPromptTemplate.from_template(chunk_process_prompt)
    agentic_chunk_chain = prompt_template | groq_model | StrOutputParser()
    return agentic_chunk_chain.invoke({'chunk': document_chunk})

def create_contextual_chunks(article_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    doc_chunks = splitter.split_text(article_text)

    contextual_chunks = []
    for chunk in doc_chunks:
        chunk_metadata = {'id': str(uuid.uuid4()), 'source': 'web_scrape'}
        print(f'Processing chunk: {chunk[:100]}')
        context = generate_chunk_context(chunk)
        contextual_chunks.append(Document(page_content=context + "\n" + chunk, metadata=chunk_metadata))

    return contextual_chunks

def index_articles(urls):
    all_docs = []
    for url in urls:
        print(f"Scraping: {url}")
        article_text = scrape_article(url)
        article_chunks = create_contextual_chunks(article_text)
        all_docs.extend(article_chunks)

    if not all_docs:
        raise ValueError("No documents scraped. Check URLs.")

    return FAISS.from_documents(all_docs, embeddings)

def generate_site_content(query, vectorstore):
    print("Searching relevant content for:", query)
    retrieved_docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    query_refinement_prompt = f"""
    You are an SEO content strategist. Given the retrieved content:
    {context}

    Generate a well-structured query that will result in an updated, high-ranking article on this topic.

    Refined Query:
    """

    refined_query_template = ChatPromptTemplate.from_template(query_refinement_prompt)
    refined_query = refined_query_template | groq_model | StrOutputParser()
    refined_query = refined_query.invoke({})

    print("Generated refined query:", refined_query)
    final_content_prompt = f"""
    You are an AI content writer generating SEO-optimized content.

    Write a detailed, engaging, and informative article based on the following query:
    {refined_query}

    Ensure the content is fresh, high-quality, and ranks well on search engines.

    Article:
    """

    final_content_template = ChatPromptTemplate.from_template(final_content_prompt)
    final_article = final_content_template | groq_model | StrOutputParser()

    return final_article.invoke({})

urls = [
    "https://www.searchenginejournal.com/artificial-intelligence-content-marketing/",
    "https://www.contentmarketinginstitute.com/2023/09/ai-content-creation/"
]
vectorstore = index_articles(urls)

site_to_update = {"www.refereneurl.com"}

query = "Latest trends in AI for content optimization and SEO best practices 2024"
final_article = generate_site_content(query, vectorstore)
print(final_article)
