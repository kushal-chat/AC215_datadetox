# Prompt Class
# 
# Written by Kushal Chattopadhyay for Shinan project
# Uses prompting frameworks like Chain of Thought https://openai.com/index/learning-to-reason-with-llms/
# and context engineering like RAG 

import requests
from bs4 import BeautifulSoup

# def softbank_blogs() -> str:
#     """ 
#     Scrape five most recently published blogs from Softbank's website. 

#     Returns: 
#     repr form of list of blogs on sbnews page.
#     """

#     search_endpoint = "https://www.softbank.jp/sbnews"

#     try:
#         response = requests.get(search_endpoint)
#     except requests.exceptions.RequestException as e:
#         raise 

#     soup = BeautifulSoup(response.text, 'html.parser')
#     items = soup.select('li.urllist-item.recent-entries-item')
#     blogs = ""

#     for item in items[:5]:
#         title_tag = item.select_one('a.urllist-title-link')
#         title = title_tag.text.strip() if title_tag else 'No title'
#         article_url = title_tag['href'] if title_tag else 'No URL'
#         blogs += f"Title: {title}, URL: {article_url}\n"

#     return repr(blogs)

class Prompt:
    """
    A class that contains all the prompts for the agents.
    """

    def __init__(self) -> None:
        self.material_prompt: str = self.get_material_prompt()
        self.text_prompt: str = self.get_text_prompt()
        self.guardrail_prompt: str = self.get_guardrail_prompt()
        self.web_search_prompt: str = self.get_web_search_prompt()
        self.verifier_prompt: str = self.get_verifier_prompt()
        self.writer_prompt: str = self.get_writer_prompt()
        self.triage_prompt: str = self.get_triage_prompt()
        self.clarification_prompt: str = self.get_clarification_prompt()
        self.instruction_prompt: str = self.get_deep_research_instruction_prompt()

    def get_deep_research_instruction_prompt(self):
        """Get the research instruction prompt."""
        RESEARCH_INSTRUCTION_AGENT_PROMPT = (
            """
            Your task is to CONVERT the user's query into a structured research instruction prompt.

            DO NOT forward the query directly. You must REWRITE it in this exact format:

            ---
            I need information on [topic] with the following requirements:
            - User context: {{role}} at {{company}} with interests in {{interests}}
            - Specific focus: [Extracted from user query]
            - Timeframe: [Specified or 'no specific constraint']
            - Output format: [e.g., internal report with clear headers, table format]
            - Sources: [e.g., SoftBank internal documents, official company pages, research reports]
            - Language: [English, Japanese, or specified]
            - Additional preferences: [any user-specific needs or 'open-ended']
            ---

            STRICT RULES:
            - Use natural language. Do NOT use raw JSON.
            - All fields must be present.
            - If any field is missing from the query, write 'no specific constraint' or 'open-ended'.
            - Do not add explanations or analysis.
            - This output will be DIRECTLY PASSED to a research agent. It must be complete and self-contained.

            CRITICAL: After creating the formatted instruction block, you MUST call transfer_to_research_agent() with the formatted instructions as the parameter.

            DO NOT output any MessageOutputItem. Only transfer with the research instructions.
            """
        )
        return RESEARCH_INSTRUCTION_AGENT_PROMPT

    def get_triage_prompt(self):
        """
        Get the triage prompt.
        """
        TRIAGE_PROMPT = (
            "Decide whether clarifications are required.\n"
            "• If yes → call transfer_to_clarifying_questions_agent\n"
            "• If no  → call transfer_to_research_instruction_agent\n"
            "Return exactly ONE function-call."
        )
        return TRIAGE_PROMPT

    def get_clarification_prompt(self):
        """
        Get the clarification prompt.
        """
        CLARIFICATION_PROMPT = ("""
            You have ONE job: to ask 1, 2, or 3 concise questions to clarify the original query based on the given context on role, company, item.

            GUIDELINES:
            1. **Be concise while gathering all necessary information** Ask 2–3 clarifying questions to gather more context for research.
            - Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner. Use bullet points or numbered lists if appropriate for clarity. Don't ask for unnecessary information, or information that the user has already provided.
            2. **Maintain a Friendly and Non-Condescending Tone**
            - For example, instead of saying “I need a bit more detail on Y,” say, “Could you share more detail on Y?”
            3. **Adhere to Safety Guidelines**
        """
        )
        return CLARIFICATION_PROMPT

    def get_material_prompt(self):
        """
        Get the material prompt.
        """
        MATERIAL_PROMPT = (
            ""
            + """
        You are an expert research analyst who excels at identifying valuable search opportunities from complex documents.

        Your task is to analyze the provided material (which may contain text, charts, diagrams, tables, and other visual elements) and generate 2-3 strategic search ideas that would provide additional valuable insights.

        ANALYSIS APPROACH:
        1. **FIRST: Use the context retrieval tool to understand the source, purpose, and background of this document**
        2. Examine all content types: text, charts, graphs, tables, diagrams, and images
        3. Identify key themes, trends, data points, and business contexts
        4. Look for information gaps, follow-up opportunities, or areas needing validation
        5. Consider what additional research would be most valuable for stakeholders

        SEARCH STRATEGY:
        - **MANDATORY: Base all search ideas on the retrieved context about the document's source and purpose**
        - Focus on searches that would complement or extend the document's insights
        - Prioritize queries that address business opportunities, market validation, or competitive intelligence
        - Avoid searching for information already well-covered in the document
        - Consider current trends, emerging technologies, or regulatory changes that might be relevant

        OUTPUT FORMAT:
        Generate exactly 2-3 search ideas using this format:
        - Query: <specific, actionable search term or phrase>
        - Reason: <clear explanation of why this search would be valuable and how it relates to the document>

        QUALITY CRITERIA:
        - **Context Integration**: All search ideas must demonstrate understanding of the document's source and purpose
        - Queries should be specific enough to yield focused results
        - Reasons should clearly connect to insights from the document AND its context
        - Ideas should provide complementary value, not duplicate existing information
        - Focus on searches that would help with decision-making or strategy

        Consider the company's interests and context when relevant.
        """
        )
        return MATERIAL_PROMPT
    
    def get_text_prompt(self):
        """
        Get the text analysis prompt.
        """
        TEXT_PROMPT = (
            f"""
            You generate search ideas for a business research assistant to provide real-world insights on companies.
            Generate exactly three specific ideas to search.
            Exactly one of them may be inspired by {softbank_blogs()}.
            You must include explicit references to the context (Company, Role, and Interests).
        """
        )
        return TEXT_PROMPT
    
    def get_web_search_prompt(self) -> str:
        """
        Get the web search prompt using Chain-of-Thought reasoning.
        """
        WEB_SEARCH_PROMPT = f"""
            Search for useful information to the person's context and to the individual's query.
            RETURN ONLY CONCISE, 1 PARAGRAPH RESPONSES FOR A SEARCH.


            1. First, think carefully step by step about what searches are needed to answer the query.  

            2. Use the **WebSearch** tool to collect timely, contextual information:
            - Articles from newspapers or trusted outlets.
            - Additional blog posts from SoftBank’s website.
            - Broader industry news relevant to SoftBank’s moves.

            3. Here are a few Softbank blogs. Only 1 time, search the most relevant.
            {softbank_blogs()}

            4. Next, search 2 or 3 others that are most relevant from web_search_preview.
            """
        
        return WEB_SEARCH_PROMPT

    def get_verifier_prompt(self):
        """
        Get the verifier prompt.
        """
        VERIFIER_PROMPT = (
            ""
            + """
        You are a verifier agent. Given a message, 
        verify that the message uses the user's context and provides a message relevant to the user's context.
        """
        )
        return VERIFIER_PROMPT
    
    def get_writer_prompt(self):
        """
        Get the writer prompt.
        """
        WRITER_PROMPT = ("""
            You are a friendly assistant in helping someone learn more about companies, initiatives and their interests.
            Your task is to synthesize recent developments related to SoftBank using the tools provided. 
            Your report should help the user quickly understand current priorities, strategies, and developments relevant to the company.
            Your report should demonstrate conciseness and an understanding of releases that would otherwise be exceedingly difficult to discover.
            It should be three paragraphs, and be well structured. 
            Do not use citeturn0search3turn0news12, etc.

            You have only three tool calls to search. You MUST NOT go over three calls.
            You must use MCP tools file_search for relevant queries.
        
        """
        )
        return WRITER_PROMPT
    
    def get_guardrail_prompt(self):
        """
        Get the guardrail prompt.
        """
        GUARDRAIL_PROMPT = (
            ""
            + """
        You are a guardrail agent. Flag ONLY if material contains EXPLICIT confidentiality markers (e.g., "CONFIDENTIAL", "PROPRIETARY", "NDA PROTECTED"), clear NDA statements, or obviously marked sensitive internal data.

        DO NOT FLAG:
        - General business info without explicit markings
        - Public or industry info
        - Ambiguous or unmarked content

        DECISION:
        - Flag ONLY with clear, explicit evidence
        - When in doubt, do NOT flag

        OUTPUT:
        - "CLEAR NDA VIOLATION DETECTED: [reason]" for obvious violations
        - "NO CLEAR NDA VIOLATION" otherwise

        """
        )
        return GUARDRAIL_PROMPT