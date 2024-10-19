import json
from textwrap import dedent
from openai import OpenAI
from dotenv import load_dotenv

from ..pydantic_classes import Paper, Summary

"""
main sections of a paper:
1. Title
2. Abstract
3. Introduction
4. Methodology
5. Results
6. Findings
"""

load_dotenv(".env")

class SummaryEngine:
    def __init__(self):
        self.client = OpenAI() 
        self.sections_to_column = None

    def map_sections_to_columns(self, sections):

        PROMPT_TEMPLATE = """ 
            your task is the map the given sections in a paper to a column.

            ## Guidelines:

            1. map each section to a column that you find most appropriate.
            2. each section MUST be mapped to a column.
            3. you can map multiple sections to a single column.
            4. You can leave columns empty if you find no section to map to it.

            ## Sections in the paper:
            {sections}

            ## Response format:

            {{
                "introduction" : [],
                "methodology" : [],
                "results" : [],
                "findings" : []
                "conclusion" : []
            }}

            Response:
        """

        prompt = dedent(PROMPT_TEMPLATE).format(sections=sections)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "you are an assistant who responds in json format"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    
    def generate_summary(self, chunks: list, column_name: str, paper_title: str, max_words: int):

        PROMPT_TEMPLATE = """ 
            Your task is to generate a summary based on the given content of the paper and the topic name based on which the summary should be generated:

            Generate a summary that at most contains {max_words} words.
            The summary should address the given topic name from the paper content.


            Paper content:
            {content}

            Topic name:
            {topic_name}

            Response format:
            ## Response Format:
            {{"response": ""}}

            Response:
        """

        content = "\n".join([chunk['page_content'] for chunk in chunks])
        prompt = dedent(PROMPT_TEMPLATE).format(content=content, topic_name=column_name, max_words=max_words)
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "you are an assistant who responds in json format"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        return json.loads(response.choices[0].message.content)['response']


    def summarize(self, paper: Paper, column: str, max_words: int = 60):
        """
            1. map section(s) to a column
            2. summarise the content as per each column. 
            3. return the summary. 
        """
        sections = []
        for chunk in paper.chunks_content:
            headers = [int(keys.split()[1]) for keys in chunk['metadata'].keys()]
            if len(headers) > 0:
                headers.sort()
                biggest_header = headers[0]
                for key, value in chunk['metadata'].items():
                    if int(key.split()[1]) == biggest_header:
                        if value not in sections:
                            sections.append(value.lower().strip())
        
        if "abstract" in sections:
            sections.remove("abstract")

        self.sections_to_column = self.map_sections_to_columns(sections)

        if column in self.sections_to_column:
            sections = self.sections_to_column[column]  
        elif column == "abstract":
            summary = self.generate_summary(
                chunks=[{"page_content": paper.abstract}],
                column_name="abstract",
                paper_title=paper.title,
                max_words=max_words
            )
            return Summary(
                paper_title=paper.title,
                summary=summary,
                column_name=column
            ) 
        else: 
            raise ValueError(f"Column name: {column} not found in the sections to column mapping.")
        
        chunks = []
        for chunk in paper.chunks_content:
            for key, value in chunk['metadata'].items():
                if value in sections:
                    chunks.append(chunk)
                    break

        summary = self.generate_summary(
            chunks=chunks, 
            column_name=column, 
            paper_title=paper.title, 
            max_words=max_words
        )
            
        return Summary(
            paper_title=paper.title,
            summary=summary,
            column_name=column
        )
            


                