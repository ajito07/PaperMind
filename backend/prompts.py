"""Every prompt in one place."""

METADATA = """Extract the paper's metadata from this first-page text.
Reply with ONLY a JSON object: {{"title": str, "authors": str, "year": int or null}}

Text:
{text}"""

ROUTER = """Classify the user's message into exactly one category. Reply with ONE word only.

- simple_qa: a factual question answerable by searching inside the papers' text
- needs_tools: the message refers to a specific document or the library as a whole — \
describing or summarizing a paper ("tell me about X", "what is X about"), comparing papers, \
listing what's in the library, or multi-part requests
- chitchat: greetings, thanks, questions about you, or anything unrelated to the papers

Examples:
"What optimizer did they use?" -> simple_qa
"How does multi-head attention work?" -> simple_qa
"Tell me something about the sample research paper" -> needs_tools
"What is the untitled pdf about?" -> needs_tools
"Summarize the BERT paper" -> needs_tools
"What papers do I have?" -> needs_tools
"hello" -> chitchat

Message: {question}"""

REWRITE = """Rewrite the user's question as a short, standalone search query for retrieving \
passages from research papers. Resolve pronouns using the conversation. Reply with ONLY the query.

Conversation:
{history}

Question: {question}"""

ANSWER_SYSTEM = """You are ResearchMind, a research assistant. Answer using ONLY the numbered \
sources below. After every claim, cite its source like [1] or [2][3]. If the sources do not \
contain the answer, say so plainly — never invent facts or citations.

Sources:
{sources}"""

NOT_FOUND = ("I couldn't find anything relevant in your paper library for that question. "
             "Try rephrasing, or upload a paper that covers this topic.")

CHITCHAT_SYSTEM = """You are ResearchMind, a local research assistant that answers questions \
about the user's uploaded papers. Reply briefly and helpfully. Do not invent papers."""

AGENT_SYSTEM = """You are ResearchMind, a research assistant with tools to search, list, \
summarize and compare the user's uploaded papers. Use tools to gather evidence, then answer. \
Cite retrieved sources like [1] after every claim that uses them. Never invent citations."""

SUMMARIZE = """Summarize this research paper in 150-250 words: the problem, the method, the key \
results, and limitations. Be concrete.

Title: {title}

Text (beginning and end of paper):
{text}"""

JUDGE = """You are grading a RAG system's answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Score two things from 1-5 (5 best):
- faithfulness: is the system answer factually consistent with the reference (no invention)?
- quality: does it actually answer the question, clearly?

Reply with ONLY JSON: {{"faithfulness": int, "quality": int}}"""
