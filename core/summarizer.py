from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

import os

def get_llm():
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
        max_retries=4,
    )

def split_transcript(transcript:str)->list:
    """Split transcript into chunks of 1000 characters with 200 character overlap"""
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(transcript)  

def summarize(transcript:str)->str:

    llm=get_llm() 

    map_prompt=ChatPromptTemplate.from_messages( 
        [
            ("system","Summarize this portion of a video transcript consisely."),
            ("human","{text}"),
        ]
    )   

    map_chain=map_prompt | llm | StrOutputParser()
    chunks=split_transcript(transcript)
    chunk_summaries=[map_chain.invoke({"text":chunk}) for chunk in chunks]

    combined="\n\n".join(chunk_summaries)

    combined_prompt=ChatPromptTemplate.from_messages(
        [
            ("system","You are an expert meeting summarizer . Combine these partial summaries"
             "into one final professional meeting summary in bullet points "),
            ("human","{text}"),
        ]
    )

    combined_chain=(
        RunnablePassthrough() | RunnableLambda(lambda x: {"text":x}) |combined_prompt |llm| StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcript:str)->str:
    llm=get_llm()

    title_chain=(
        RunnablePassthrough() | RunnableLambda(lambda x: {"text":x}) | 
        ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript,generate a short professional meeting title"
                "(max 7 words) . Only retrun the title ,nothing more",
         ),
            ("human","{text}"),
        ])
        |llm
        |StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000])  # Use only the first 2000 characters for title generation