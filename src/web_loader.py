import requests
from bs4 import BeautifulSoup
import re
import html2text
from langchain_text_splitters import MarkdownHeaderTextSplitter

class WebLoader:
    def clean_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove <head>, <annotation-xml>, and <semantics> tags (including their content)
        for tag in soup(['head', 'annotation-xml', 'semantics', 'footer', 'script']):
            tag.decompose()
        
        # Remove <div> and <span> tags but keep their content
        for tag in soup.find_all(['div', 'span', '!DOCTYPE html', 'article', 'html', 'body']):
            # Unwrap the tag (replace the tag with its content)
            tag.unwrap()
        
        # Remove all attributes except for 'href' in <a> and 'src' in <img>
        for tag in soup.find_all(True):  # True finds all tags
            if tag.name == 'a':
                attrs_to_keep = {'href'}
            elif tag.name == 'img':
                attrs_to_keep = {'src'}
            else:
                attrs_to_keep = set()
                
            for attr in list(tag.attrs):
                if attr not in attrs_to_keep:
                    del tag[attr]
        
        cleaned_html = str(soup)
        cleaned_html = re.sub(r'\n{2,}', '\n', cleaned_html)
        cleaned_html = re.sub(r'•\n', '', cleaned_html)
        return cleaned_html
    
    def chunk_content(self, md_content):
        headers_to_split_on = [
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        chunks = splitter.split_text(md_content)
        chunks = [chunk.model_dump() for chunk in chunks]

        return chunks
    

    def load_paper(self, url):
        response = requests.get(url)
        response.raise_for_status()
        html_content = response.text

        cleaned_html = self.clean_html(html_content)
        markdown_content = html2text.html2text(cleaned_html)
        chunks = self.chunk_content(markdown_content)

        return chunks


if __name__ == '__main__':
    loader = WebLoader()
    url = 'https://ar5iv.labs.arxiv.org/html/2103.10360'
    chunks = loader.load_paper(url)
    
    for c in chunks:
        print(c)
        print()