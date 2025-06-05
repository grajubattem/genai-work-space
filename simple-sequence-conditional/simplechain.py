from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini model
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

# Use Gemini (ChatGoogleGenerativeAI)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", convert_system_message_to_human=True)

parser = StrOutputParser()

chain = prompt | model | parser

# invoke function to run the chain
result = chain.invoke({'topic': 'cricket'})
print(result)

# visualize the chain
chain.get_graph().print_ascii()
