LLM_CORRUPTER_PROMPT = (
    "You are an expert text editor. Your task is to rewrite a news article in plain English. "
    "Make only small and natural changes, such as replacing a few words with synonyms or slightly rephrasing sentences. "
    "Keep the same overall meaning, facts, and tone of the original. "
    "Ensure the rewritten text stays very close in meaning to the original — at least 85% semantically similar. "
    "Do not add or remove information, and do not explain your edits. "
    "Write only the rewritten news text, nothing else.\n\n"
    "Example:\n"
    "Original: Donald Trump publicly criticized the Pope during his speech on foreign policy.\n"
    "Rewritten: The U.S. president publicly criticized the Pope during a foreign policy address.\n\n"
    "Now rewrite the following news text:"
)