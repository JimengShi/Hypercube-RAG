import os
import json
import argparse
from openai import OpenAI

# ================ Setup & Client ================
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ================ Unified Prompt Logic ================
def get_unified_extraction_prompt(sentence):
    """
    Combines all extraction logic into a single instructions block.
    """
    dimensions_desc = """
    1. date: specific time, period, duration, hour, minute, day, week, month, year, and descriptive phrases/terms such as six months, half hour, one decade.
    2. location: specific geographic locations, nationality, city, state, province, zipcodes.
    3. person: specific names of people, males, females, children, adults and the descriptive phrases/terms such as person, women, adult, and the role/occupration names such as father, parents, patient, doctor, lawer, consultant, disclosing party and receiving party. 
    4. quantity: numbers, percentage, ratio, rates, fractions, ages, weights, finance/money, and descriptive phrases/terms such as such as increase, decrease, decline.
    5. organization_company: organizations, programs, projects, journals, department, and university-related entities/phrases, company: names of companies, such as, "Deloitte Real Estate", "Company A", "CopAcc", "ToP Mentors", "DoiT", "ICN", "FNHA", and entities/objects have involved with agreement.
    6. money_finance: money, finance, transactions, grant, fund, and asset-related entities/phrases/terms.
    7. law_agreement_regulation: law, agreement, regulation, rule, policy-based entities/phrases/terms, e.g., "non-disclosure agreement", "law", "regulation", "judicial".
    8. information: confidential information, technical information, public information-related, news-based entities/phrases/terms, e.g., "confidential information", "technical information", "public information", "news".
    """

    prompt = f"""
    Extract entities from the sentence below based on these 9 dimensions:
    {dimensions_desc}. You can expand the dimensional values of each dimension if you think it is related.

    Guidelines:
    - Count the frequency of each entity found.
    - Use lowercases for the extracted terms.
    - Return a JSON object where keys are the dimension names and values are dictionaries of {{ "entity": frequency }}.
    - If a dimension has no entities, return an empty dictionary for that key.

    Sentence: "{sentence}"
    """
    return prompt

# ================ Main Processing ================
def parse_args():
    parser = argparse.ArgumentParser(description="Multi-dimensional Entity Extraction")
    parser.add_argument("--data", type=str, required=True, help="Dataset name (e.g., scifact)")
    parser.add_argument("--input", type=str, required=False, help="Path to input text file")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define the dimensions we expect to see in the output
    dims = [
        "date", "location", "person", "quantity", 
        "organization_company", "money_finance", 
        "law_agreement_regulation", "information"
    ]

    # Create output directory
    output_dir = f"hypercube_new/{args.data}"
    os.makedirs(output_dir, exist_ok=True)

    if args.data == "scifact":
        file_path = "corpus/scifact/pubmed_abstract.txt"
    elif args.data == "legalbench":
        file_path = "corpus/legalbench/contractnli.txt"
    elif args.data == "hurricane":
        file_path = "corpus/hurricane/SciDCC-Hurricane.txt"

    # Read input lines
    with open(file_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for idx, line in enumerate(lines):
        print(f"Processing line {idx + 1}/{len(lines)}...")
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": get_unified_extraction_prompt(line)}],
                response_format={"type": "json_object"}, # Ensures valid JSON output
                temperature=0,
            )

            # Parse the unified response
            full_result = json.loads(response.choices[0].message.content)

            # Distribute results into their respective dimension files
            for d in dims:
                # Get the specific dict for this dimension, default to empty if missing
                dim_data = full_result.get(d, {})
                
                output_file = f"{output_dir}/{d}.txt"
                with open(output_file, "a") as f_out:
                    f_out.write(json.dumps(dim_data) + "\n")

        except Exception as e:
            print(f"Error on line {idx+1}: {e}")
            # Log empty dicts for all dimensions on failure to keep line alignment
            for d in dims:
                with open(f"{output_dir}/{d}.txt", "a") as f_out:
                    f_out.write(json.dumps({}) + "\n")

    print(f"Extraction complete. Files saved in {output_dir}")

if __name__ == "__main__":
    main()