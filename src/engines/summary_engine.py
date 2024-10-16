from textwrap import dedent
from openai import OpenAI
from dotenv import load_dotenv

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


    def map_section_to_column(self, columns):
        # extract sections
        pass


    def summarize(self, content, columns, max_tokens=100):
        """
            1. map section(s) to a column
            2. summarise the content as per each column. 
            3. return the summary. 
        """