from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from enum import Enum

class Document(BaseModel):
    uuid: int
    title: str
    abstract: str
    categories: list[str]
    authors: list[str]
    create_date: datetime
    update_date: datetime
    paper_url: str
    html_url: str
    paper_id: str
    doi: Optional[str] = None
    report_no: Optional[str] = None
    journal_ref: Optional[str] = None
    license: Optional[str] = None
    comments: Optional[str] = None
    metadata: Optional[dict] = None

class DateRange(BaseModel):
    start_date: str
    end_date: str


arxiv_categories = [
    'Machine Learning',
    'Artificial Intelligence',
    'Machine Learning (Statistics)',
    'Computer Vision and Pattern Recognition',
    'Computation and Language',
    'Robotics',
    'Cryptography and Security',
    'Information Theory',
    'Neural and Evolutionary Computing',
    'Image and Video Processing (Electrical Engineering and Systems Science)',
    'Optimization and Control',
    'Information Retrieval',
    'Signal Processing (Electrical Engineering and Systems Science)',
    'Audio and Speech Processing (Electrical Engineering and Systems Science)',
    'Computers and Society',
    'Sound',
    'Human-Computer Interaction',
    'Systems and Control',
    'Numerical Analysis',
    'Distributed, Parallel, and Cluster Computing',
    'Systems and Control (Electrical Engineering and Systems Science)',
    'Social and Information Networks',
    'Mathematics of Computing',
    'Software Engineering',
    'Data Structures and Algorithms',
    'Methodology (Statistics)',
    'Statistics (Mathematics)',
    'Statistics Theory',
    'Networking and Internet Architecture',
    'Logic in Computer Science',
    'Computer Science and Game Theory',
    'Applications (Statistics)',
    'Quantitative Methods (Quantitative Biology)',
    'Databases',
    'Quantum Physics',
    'Multimedia',
    'Neurons and Cognition (Quantitative Biology)',
    'Graphics',
    'Computational Engineering, Finance, and Science',
    'Computational Physics',
    'Computation (Statistics)',
    'Architecture',
    'Probability',
    'Programming Languages',
    'Biomolecules (Quantitative Biology)',
    'Computational Complexity',
    'Data Analysis, Statistics and Probability',
    'Dynamical Systems',
    'Disordered Systems and Neural Networks',
    'Materials Science'
]

ArxivCategories = Enum('ArxivCategories', {category.replace(' ', '_').upper(): category for category in arxiv_categories})


class Query(BaseModel):
    question: str
    categories: Optional[list[ArxivCategories]] = None # type: ignore
    authors: Optional[list[str]] = None
    date_range: Optional[DateRange] = None