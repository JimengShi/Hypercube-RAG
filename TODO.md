

**I. Short-Term Research Questions**

1. **Taxonomy Extraction**

   * How can we derive a preliminary hierarchical theme taxonomy from the SciQA corpus?
2. **Query-to-Taxonomy Mapping**

   * Which techniques (e.g., keyword matching vs. embedding clustering) reliably place individual SciQA queries into our theme hierarchy?

---

**II. Exploration Workflow**

1. **Data Preparation**

   * Sample 100–200 SciQA queries and pull their associated document contexts.
2. **Corpus-Driven Taxonomy Construction**

   * Apply a topic-modeling or hierarchical-clustering method to the document texts.
   * Manually inspect the resulting clusters and label them into two levels (Level 1: broad domain; Level 2: subdomain).
3. **Mapping Prototype**

   * **Keyword-Based:** Build a simple rule matcher that looks for domain/subdomain keywords in queries.
   * **Embedding-Based:** Compute query embeddings (e.g., with a pre-trained BERT) and assign each query to the nearest cluster centroid (cosine similarity).
4. **Validation & Iteration**

   * For \~20 held-out queries, check if the assigned theme (by each method) matches human judgment.
   * Refine cluster labels or mapping thresholds based on misassignments.

---

**III. Expected Deliverables by Week’s End**

1. **Draft Two-Level Taxonomy**

   * A document listing Level 1 → Level 2 theme pairs (e.g., Physics → Theoretical Physics).
2. **Mapping Prototype Notebook**

   * Code showcasing both the keyword and embedding approaches, with sample assignments.
3. **Assignment Accuracy Table**

   * A short table (20 queries × 2 methods) comparing automated vs. human theme assignments.
4. **Next Steps Brief**

   * Recommendations on scaling: e.g., automating label refinement or extending to more levels.
