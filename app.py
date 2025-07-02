from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

sonnet = init_chat_model("anthropic.claude-3-5-sonnet-20240620-v1:0", model_provider="bedrock_converse")
titan = init_chat_model("amazon.titan-tg1-large", model_provider="bedrock_converse")


messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]
for token in sonnet.stream(messages):
    print(token.content, end="|")

system_template = "Translate the following from English into {language}"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})
print(prompt)
