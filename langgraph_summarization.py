import warnings
warnings.filterwarnings("ignore")

import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

import os
import asyncio

from langchain_community.document_loaders import TextLoader

Costs={
    'gpt-4o-mini':{'input':0.15e-6,'output':0.6e-6},
    'gpt-3.5-turbo-1106':{'input':1e-6,'output':2e-6}
}

token_max=1000
model='gpt-4o-mini'
llm = ChatOpenAI(model=model,seed=41) 

def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)

# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str
    factchecking: str

# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str


# Here we generate a summary, given a document
async def generate_summary(state: SummaryState):
    map_prompt = ChatPromptTemplate.from_messages([("system", "Write a concise summary of the following, that maintains all quantitative details, filters out all comments not strictly related to science, doesn't use the words 'speaker' nor 'summary' nor synonyms of these words, and does not use emotional language. \\n\\n{context}")])
    map_chain = map_prompt | llm 
    response = await map_chain.ainvoke(state["content"])
    return {"summaries": [response.content]} 


# Here we define the logic to map out over the documents
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects: each `Send` object consists of the name of a node in the graph as well as the state to send to that node
    to_send=[Send("generate_summary", {"content": content}) for content in state["contents"]]
    print(f"Will branch out on {len(to_send)} parallel summarisations.")
    return to_send


def collect_summaries(state: OverallState):
    collapsed_summ=[Document(summary) for summary in state["summaries"]]
    print(f"{len(collapsed_summ)} collapsed summaries.")
    return {"collapsed_summaries": collapsed_summ}

# Add node to collapse summaries
async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(state["collapsed_summaries"], length_function, token_max)
    results = []
    print(f"Split into {len(doc_lists)}")
    reduce_prompt = ChatPromptTemplate.from_messages([("system", " The following is a set of summaries:\\n\\n{docs}\\n\\nTake these and distill it into a final, consolidated summary of the main themes.")])
    reduce_chain = reduce_prompt | llm 
    for doc_list in doc_lists:
        #print(doc for doc in doc_list)
        results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))
    #print("FINSI")
    return {"collapsed_summaries": results}


# This represents a conditional edge in the graph that determines if we should collapse the summaries or not
def should_collapse(state: OverallState) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        print(f"Must collapse: {num_tokens} > {token_max}")
        #return "collapse_summaries" # For now, collapse_summaries is disabled, so effectively this won't be a conditional edge
        return "generate_final_summary"
    else:
        print("No need for collapse.")
        return "generate_final_summary"

# Here we will generate the final summary
async def generate_final_summary(state: OverallState):
    reduce_prompt = ChatPromptTemplate.from_messages([("system", " The following is a set of summaries:\\n\\n{docs}\\n\\nTake these and distill it into a final, consolidated summary of the main themes.")])
    reduce_chain = reduce_prompt | llm 
    response = await reduce_chain.ainvoke(state["collapsed_summaries"])
    return {"final_summary": response.content} 

def summarizer():
    # Construct the graph
    # Nodes:
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)  # same as before
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    # Edges:
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    app = graph.compile()
    image_path = app.get_graph().draw_mermaid_png()
    with open('FlowChart_summarizer.png', 'wb') as f:
        f.write(image_path)
    return app

def summarizer_doc(split_docs, token_max=1000):
    app=summarizer()
    # Execute the graph on input document
    async def loop(split_docs):
        async for step in app.astream({"contents": [doc.page_content for doc in split_docs]},{"recursion_limit": 10}):
            if 'generate_final_summary' in step.keys(): 
                text_summary='\n'.join([sentence+'.' for sentence in step['generate_final_summary']['final_summary'].split('.')])
        return text_summary

    text_summary = asyncio.run(loop(split_docs))
    return text_summary

