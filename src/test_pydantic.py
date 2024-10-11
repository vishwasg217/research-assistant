from pydantic_classes import Query, DateRange

# example
transformed_query = Query(
    question="How to train a machine learning model?",
    categories=["Machine Learning"],
    authors=["John Doe", "Jane Doe"],
    date_range=DateRange(start_date="2020-01-01", end="2023-01-01")
)

print(transformed_query)
print([cat.value for cat in transformed_query.categories])