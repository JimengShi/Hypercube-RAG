# TODO

import requests

# SPARQL query to fetch scholarly articles
query = """
PREFIX schema: <http://schema.org/>
SELECT ?paper ?title
WHERE {
  ?paper a schema:ScholarlyArticle ;
         schema:name ?title .
}
LIMIT 10
"""

# Endpoint URL (can use either triplestore or sparql)
url = "https://www.orkg.org/orkg/sparql"

resp = requests.post(
    url,
    data=query.encode("utf-8"),  # Query text as byte stream
    headers={
        "Content-Type": "application/sparql-query",           # Key: tells server that body is SPARQL query
        "Accept":        "application/sparql-results+json"    # Request JSON format results
    },
    timeout=30
)
resp.raise_for_status()

# Parse the response
data = resp.json()
for row in data["results"]["bindings"]:
    uri   = row["paper"]["value"]
    title = row["title"]["value"]
    print(f"{uri}\t– {title}")
