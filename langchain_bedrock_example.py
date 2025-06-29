"""
LangChain with Amazon Bedrock Example
This file demonstrates various LangChain concepts using Claude 3.5 Sonnet from Amazon Bedrock.

What is LangChain?
LangChain is a framework for developing applications powered by large language models (LLMs).
It provides tools for:
- Connecting to different LLM providers
- Creating prompt templates
- Chaining multiple operations together
- Building conversational agents
- Managing memory and context
- Processing and analyzing documents
"""

import os
import boto3
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
import json
import time

# Load environment variables from .env file
load_dotenv()

# LangSmith setup - this will automatically be enabled if the environment variables are set
print("LangSmith Configuration:")
print(f"   LANGSMITH_TRACING: {os.getenv('LANGSMITH_TRACING', 'Not set')}")
print(f"   LANGSMITH_PROJECT: {os.getenv('LANGSMITH_PROJECT', 'Not set')}")
print(f"   LANGSMITH_ENDPOINT: {os.getenv('LANGSMITH_ENDPOINT', 'Not set')}")
print(f"   LANGSMITH_API_KEY: {'Set' if os.getenv('LANGSMITH_API_KEY') else 'Not set'}")
print()

if os.getenv('LANGSMITH_TRACING') == 'true':
    print("LangSmith tracing is enabled! Visit https://smith.langchain.com to view traces.")
    print(f"   Project: {os.getenv('LANGSMITH_PROJECT')}")
    print()
else:
    print("LangSmith tracing is not enabled.")
    print()


def demo_1_basic_llm():
    """
    Demo 1: Basic LLM Setup
    This shows how to initialize an LLM with LangChain and make a simple call.
    """
    print("=== Demo 1: Basic LLM Setup ===")
    
    # Initialize the LLM
    llm = ChatBedrock(
        model_id="amazon.titan-tg1-large",
        model_kwargs={
            "max_tokens": 1000,
            "temperature": 0.7
        }
    )
    
    # Simple direct call
    response = llm.invoke("What is artificial intelligence in one sentence?")
    print(f"Direct LLM Response: {response.content}")
    print()


def demo_2_prompt_templates():
    """
    Demo 2: Prompt Templates
    Prompt templates allow you to create reusable, parameterized prompts.
    """
    print("=== Demo 2: Prompt Templates ===")
    
    llm = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs={"max_tokens": 500, "temperature": 0.5}
    )
    
    # Create a prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that explains concepts clearly."),
        ("human", "Explain {concept} to someone who is {experience_level}.")
    ])
    
    # Create a chain
    chain = prompt | llm | StrOutputParser()
    
    # Use the chain with different inputs
    result = chain.invoke({
        "concept": "machine learning", 
        "experience_level": "completely new to programming"
    })
    
    print("Prompt Template Result:")
    print(result)
    print()


def demo_3_chains():
    """
    Demo 3: Chains
    Chains allow you to combine multiple operations in sequence.
    """
    print("=== Demo 3: Chains ===")
    
    llm = ChatBedrock(
        # model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_id="amazon.titan-tg1-large",
        model_kwargs={"max_tokens": 800, "temperature": 0.6}
    )
    
    # Create multiple prompt templates
    topic_prompt = ChatPromptTemplate.from_template(
        "Generate a creative topic about {subject} for a blog post. Just return the topic title."
    )
    
    outline_prompt = ChatPromptTemplate.from_template(
        "Create a detailed outline for a blog post with this title: {topic}"
    )
    
    # Create chains
    topic_chain = topic_prompt | llm | StrOutputParser()
    outline_chain = outline_prompt | llm | StrOutputParser()
    
    # Simplified approach - run chains sequentially
    subject = "artificial intelligence"
    
    # Step 1: Generate topic
    topic = topic_chain.invoke({"subject": subject})
    print(f"Generated topic: {topic}")
    
    # Step 2: Generate outline
    outline = outline_chain.invoke({"topic": topic})
    
    result = {"topic": topic, "outline": outline}
    
    print("Chain Result:")
    print(f"Topic: {result['topic']}")
    print(f"Outline:\n{result['outline']}")
    print()


def demo_4_conversation_memory():
    """
    Demo 4: Conversation Memory
    This demonstrates how to maintain context across multiple interactions.
    """
    print("=== Demo 4: Conversation Memory ===")
    
    llm = ChatBedrock(
        # model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_id="amazon.titan-tg1-large",
        model_kwargs={"max_tokens": 500, "temperature": 0.7}
    )
    
    # Initialize memory
    memory = ConversationBufferMemory(return_messages=True)
    
    # Create a prompt template that includes chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Keep track of the conversation context."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}")
    ])
    
    # Create chain with memory
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(
            "Previous conversation:\n{history}\n\nHuman: {input}\nAssistant:"
        ),
        memory=memory,
        verbose=True
    )
    
    # Simulate a conversation
    print("Starting conversation...")
    
    response1 = chain.invoke({"input": "My name is Alex and I love Python programming."})
    print(f"Response 1: {response1['text']}")
    
    response2 = chain.invoke({"input": "What programming language did I mention?"})
    print(f"Response 2: {response2['text']}")
    
    response3 = chain.invoke({"input": "What's my name?"})
    print(f"Response 3: {response3['text']}")
    print()


def demo_5_custom_chain():
    """
    Demo 5: Custom Chain with Multiple Steps
    This shows how to create more complex chains with custom logic.
    """
    print("=== Demo 5: Custom Chain with Multiple Steps ===")
    
    llm = ChatBedrock(
        # model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_id="amazon.titan-tg1-large",
        model_kwargs={"max_tokens": 300, "temperature": 0.5}
    )
    
    # Define multiple processing steps
    def analyze_sentiment(text):
        sentiment_prompt = ChatPromptTemplate.from_template(
            "Analyze the sentiment of this text and respond with just: POSITIVE, NEGATIVE, or NEUTRAL\n\nText: {text}"
        )
        sentiment_chain = sentiment_prompt | llm | StrOutputParser()
        return sentiment_chain.invoke({"text": text}).strip()
    
    def generate_response(text, sentiment):
        response_prompt = ChatPromptTemplate.from_template(
            "The user said: '{text}'\nThe sentiment is: {sentiment}\n\n"
            "Generate an appropriate response that acknowledges the sentiment."
        )
        response_chain = response_prompt | llm | StrOutputParser()
        return response_chain.invoke({"text": text, "sentiment": sentiment})
    
    # Test the custom chain
    user_input = "I'm really frustrated with this new software update!"
    
    print(f"User input: {user_input}")
    
    # Step 1: Analyze sentiment
    sentiment = analyze_sentiment(user_input)
    print(f"Detected sentiment: {sentiment}")
    
    # Step 2: Generate appropriate response
    response = generate_response(user_input, sentiment)
    print(f"Generated response: {response}")
    print()


def demo_6_few_shot_prompting():
    """
    Demo 6: Few-Shot Prompting
    This demonstrates how to provide examples to guide the model's behavior.
    """
    print("=== Demo 6: Few-Shot Prompting ===")
    
    llm = ChatBedrock(
        # model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_id="amazon.titan-tg1-large",
        model_kwargs={"max_tokens": 200, "temperature": 0.3}
    )
    
    # Create a few-shot prompt template
    few_shot_prompt = ChatPromptTemplate.from_template("""
You are a code comment generator. Given a function, generate a clear, concise comment.

Example 1:
Code: def add(a, b): return a + b
Comment: # Adds two numbers and returns the result

Example 2:
Code: def get_user_name(user_id): return database.query(user_id).name
Comment: # Retrieves and returns the name of a user by their ID from the database

Now generate a comment for this code:
Code: {code}
Comment:""")
    
    chain = few_shot_prompt | llm | StrOutputParser()
    
    # Test with different code examples
    test_codes = [
        "def calculate_tax(income, rate): return income * rate",
        "def send_email(recipient, subject, body): email_client.send(recipient, subject, body)"
    ]
    
    for code in test_codes:
        result = chain.invoke({"code": code})
        print(f"Code: {code}")
        print(f"Generated comment: {result.strip()}")
        print()


def demo_7_structured_output():
    """
    Demo 7: Structured Output
    This shows how to get structured data from the LLM.
    """
    print("=== Demo 7: Structured Output ===")
    
    llm = ChatBedrock(
        model_id="amazon.titan-tg1-large",
        model_kwargs={"max_tokens": 500, "temperature": 0.2}
    )
    
    # Create a prompt for structured output
    structured_prompt = ChatPromptTemplate.from_template("""
Extract the following information from the text and return it as a JSON object:
- name: person's name
- age: person's age (if mentioned)
- occupation: person's job (if mentioned)
- location: where they live (if mentioned)
- interests: list of interests or hobbies

Text: {text}

Return only valid JSON:""")
    
    chain = structured_prompt | llm | StrOutputParser()
    
    sample_text = """
    Hi, I'm Sarah Johnson, a 28-year-old software engineer living in Seattle. 
    I love hiking, reading science fiction novels, and playing the guitar. 
    I've been working in tech for about 5 years now.
    """
    
    result = chain.invoke({"text": sample_text})
    print(f"Input text: {sample_text}")
    print(f"Structured output: {result}")
    
    # Try to parse as JSON to verify structure
    try:
        parsed = json.loads(result)
        print("Successfully parsed as JSON!")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print("Warning: Output is not valid JSON")
    print()


def display_langchain_flow_diagrams():
    """
    Display visual diagrams of LangChain processes and flows
    """
    print("LangChain Flow Diagrams")
    print("=" * 50)
    
    # Basic LLM Flow
    print("\n1. Basic LLM Flow:")
    print("""
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │    Input    │───▶│     LLM     │───▶│   Output    │
    │   (Text)    │    │  (Claude)   │    │  (Response) │
    └─────────────┘    └─────────────┘    └─────────────┘
    """)
    
    # Prompt Template Flow
    print("\n2. Prompt Template Flow:")
    print("""
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Variables   │───▶│   Prompt    │───▶│     LLM     │───▶│   Output    │
    │{name: "AI"} │    │  Template   │    │  (Claude)   │    │  (Response) │
    │{task: "..."}│    │"Explain {}" │    │             │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
    """)
    
    # Chain Flow
    print("\n3. Chain Flow (Sequential Operations):")
    print("""
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Input     │───▶│  Prompt 1   │───▶│    LLM 1    │───▶│  Output 1   │
    │             │    │             │    │             │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                     │
                                                                     ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │Final Output │◀───│    LLM 2    │◀───│  Prompt 2   │◀───│ (as input)  │
    │             │    │             │    │             │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
    """)
    
    # Memory Flow
    print("\n4. Memory Flow (Conversation Context):")
    print("""
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ User Input  │───▶│   Memory    │───▶│ Full Context│───▶│     LLM     │
    │"What's my   │    │   Store     │    │User + History│   │  (Claude)   │
    │ name?"      │    │             │    │             │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                              ▲                                      │
                              │                                      ▼
                              │             ┌─────────────┐    ┌─────────────┐
                              └─────────────│   Update    │◀───│ LLM Response│
                                            │   Memory    │    │             │
                                            └─────────────┘    └─────────────┘
    """)
    
    # RAG Flow
    print("\n5. RAG Flow (Retrieval Augmented Generation):")
    print("""
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Query     │───▶│ Vector DB   │───▶│ Relevant    │
    │"How to...?" │    │   Search    │    │ Documents   │
    └─────────────┘    └─────────────┘    └─────────────┘
                                                  │
                                                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Final       │◀───│     LLM     │◀───│ Query +     │
    │ Answer      │    │  (Claude)   │    │ Context     │
    └─────────────┘    └─────────────┘    └─────────────┘
    """)
    
    # Agent Flow
    print("\n6. Agent Flow (Tool Usage):")
    print("""
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ User Query  │───▶│    Agent    │───▶│ Plan Tools  │
    │"Calculate   │    │   (LLM)     │    │ to Use      │
    │ 15 * 23"    │    │             │    │             │
    └─────────────┘    └─────────────┘    └─────────────┘
                                                  │
                                                  ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Final       │◀───│ Agent       │◀───│ Tool        │
    │ Response    │    │ Response    │    │ Execution   │
    └─────────────┘    └─────────────┘    └─────────────┘
    """)
    
    # Complete LangChain Pipeline
    print("\n7. Complete LangChain Pipeline:")
    print("""
    ┌─────────────┐
    │ Raw Input   │
    │   Data      │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐    ┌─────────────┐
    │ Data        │───▶│ Document    │
    │ Loading     │    │ Processing  │
    └─────────────┘    └─────────────┘
           │                   │
           ▼                   ▼
    ┌─────────────┐    ┌─────────────┐
    │ Vector      │    │ Prompt      │
    │ Storage     │    │ Templates   │
    └──────┬──────┘    └──────┬──────┘
           │                   │
           ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Retrieval   │───▶│ LLM Chain   │───▶│ Output      │
    │ System      │    │ Processing  │    │ Parser      │
    └─────────────┘    └─────────────┘    └─────────────┘
           ▲                   ▲                   │
           │                   │                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ User Query  │    │ Memory      │    │ Structured  │
    │             │    │ Management  │    │ Response    │
    └─────────────┘    └─────────────┘    └─────────────┘
    """)
    
    print("\n8. LangChain Component Interaction:")
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    LangChain Framework                      │
    │                                                             │
    │  ┌────────────┐  ┌────────────┐  ┌────────────┐             │
    │  │   Models   │  │  Prompts   │  │   Memory   │             │
    │  │  (LLMs)    │  │ Templates  │  │   Store    │             │
    │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘             │
    │        │               │               │                    │
    │        └───────────────┼───────────────┘                    │
    │                        │                                    │
    │  ┌────────────┐  ┌─────▼──────┐  ┌────────────┐             │
    │  │   Tools    │  │   Chains   │  │  Agents    │             │
    │  │ Functions  │  │ Workflows  │  │ Decision   │             │
    │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘             │
    │        │               │               │                    │
    │        └───────────────┼───────────────┘                    │
    │                        │                                    │
    │  ┌────────────┐  ┌─────▼──────┐  ┌────────────┐             │
    │  │ Retrievers │  │  Output    │  │Callbacks/  │             │
    │  │(Vector DB) │  │  Parsers   │  │Monitoring  │             │
    │  └────────────┘  └────────────┘  └────────────┘             │
    └─────────────────────────────────────────────────────────────┘
    """)


def demo_8_flow_visualization():
    """
    Demo 8: Live Flow Visualization
    This shows the actual flow of data through a LangChain pipeline step by step.
    """
    print("=== Demo 8: Live Flow Visualization ===")
    
    llm = ChatBedrock(
        model_id="amazon.titan-tg1-large",
        model_kwargs={"max_tokens": 300, "temperature": 0.7}
    )
    
    print("Watch the LangChain pipeline in action:")
    print("-" * 45)
    
    # Step 1: Input Processing
    print("\nStep 1: Input Processing")
    user_input = "I need help with Python loops"
    print(f"   Raw Input: '{user_input}'")
    time.sleep(1)
    
    # Step 2: Prompt Template
    print("\nStep 2: Prompt Template Application")
    prompt_template = ChatPromptTemplate.from_template(
        "You are a Python tutor. Help explain: {topic}. Keep it concise and practical."
    )
    print(f"   Template: 'You are a Python tutor. Help explain: {{topic}}...'")
    print(f"   Variables: {{topic: '{user_input}'}}")
    time.sleep(1)
    
    # Step 3: LLM Processing
    print("\nStep 3: LLM Processing")
    print("   Sending to Amazon Bedrock Titan...")
    chain = prompt_template | llm | StrOutputParser()
    
    try:
        response = chain.invoke({"topic": user_input})
        print("   LLM Response Generated")
        time.sleep(1)
        
        # Step 4: Output Processing
        print("\nStep 4: Output Processing & Parsing")
        print("   Parsing string output...")
        print("   Formatting response...")
        time.sleep(1)
        
        # Step 5: Final Result
        print("\nStep 5: Final Result")
        print("   Delivering to user:")
        print(f"   {response}")
        
        # Flow Summary
        print("\nFlow Summary:")
        print("   Input → Template → LLM → Parser → Output")
        print("   Pipeline completed successfully!")
        
    except Exception as e:
        print(f"   Error in pipeline: {e}")
    
    print()


def demo_9_langsmith_tracing():
    """
    Demo 9: LangSmith Tracing and Monitoring
    This demonstrates how LangSmith tracks your LangChain applications.
    """
    print("=== Demo 9: LangSmith Tracing & Monitoring ===")
    
    if os.getenv('LANGSMITH_TRACING') != 'true':
        print("LangSmith tracing is not enabled.")
        print("To enable tracing, make sure your .env file contains:")
        print("   LANGSMITH_TRACING=true")
        print("   LANGSMITH_API_KEY=<your-api-key>")
        print("   LANGSMITH_PROJECT=llm-app-test")
        print()
        return
    
    print("Demonstrating LangSmith tracing with a complex chain...")
    print("   This will be automatically tracked in LangSmith!")
    print()
    
    llm = ChatBedrock(
        model_id="amazon.titan-tg1-large",
        model_kwargs={"max_tokens": 400, "temperature": 0.7}
    )
    
    # Create a multi-step chain that will be well-traced by LangSmith
    print("Step 1: Creating a complex chain with multiple components")
    
    # Analysis prompt
    analysis_prompt = ChatPromptTemplate.from_template(
        "Analyze this user request and extract the main intent: {user_input}"
    )
    
    # Response prompt
    response_prompt = ChatPromptTemplate.from_template(
        "Based on this analysis: {analysis}\n\nGenerate a helpful response to: {original_request}"
    )
    
    # Create individual chains
    analysis_chain = analysis_prompt | llm | StrOutputParser()
    response_chain = response_prompt | llm | StrOutputParser()
    
    # Test input
    user_request = "I'm having trouble understanding how to use decorators in Python. Can you help?"
    
    print(f"Step 2: Processing user request: '{user_request}'")
    print("   (Check LangSmith to see the detailed trace!)")
    
    try:
        # Step 1: Analyze the request
        analysis = analysis_chain.invoke({"user_input": user_request})
        print(f"   Analysis complete: {analysis[:100]}...")
        
        # Step 2: Generate response based on analysis
        final_response = response_chain.invoke({
            "analysis": analysis,
            "original_request": user_request
        })
        
        print("Step 3: Final response generated")
        print(f"   Response: {final_response[:200]}...")
        
        print("\nLangSmith Benefits Demonstrated:")
        print("   Automatic tracing of all LLM calls")
        print("   Input/output logging for each step")
        print("   Timing and performance metrics")
        print("   Error tracking and debugging info")
        print("   Chain visualization and flow analysis")
        
        print(f"\nView this trace at: https://smith.langchain.com/")
        print(f"   Project: {os.getenv('LANGSMITH_PROJECT')}")
        
    except Exception as e:
        print(f"   Error in traced chain: {e}")
        print("   (This error would also be captured in LangSmith!)")
    
    print()


def demo_10_langsmith_features():
    """
    Demo 10: Advanced LangSmith Features
    This shows more advanced LangSmith monitoring capabilities.
    """
    print("=== Demo 10: Advanced LangSmith Features ===")
    
    if os.getenv('LANGSMITH_TRACING') != 'true':
        print("LangSmith tracing is not enabled. Enable it to see advanced features.")
        print()
        return
    
    from langsmith import Client
    
    try:
        # Initialize LangSmith client
        client = Client()
        project_name = os.getenv('LANGSMITH_PROJECT', 'llm-app-test')
        
        print("LangSmith Advanced Features:")
        print(f"   Connected to project: {project_name}")
        print()
        
        print("What LangSmith tracks automatically:")
        features = [
            "Chain execution flows and dependencies",
            "Latency and performance metrics for each step",
            "Token usage and cost tracking",
            "Error rates and failure analysis",
            "Model performance comparisons",
            "Retry attempts and success rates",
            "Prompt template usage patterns",
            "Input/output data for debugging"
        ]
        
        for feature in features:
            print(f"   {feature}")
        
        print("\nLangSmith Dashboard Features:")
        dashboard_features = [
            "Performance analytics and trends",
            "Trace search and filtering",
            "Custom metrics and monitoring",
            "Real-time debugging and inspection",
            "A/B testing for prompt variations",
            "Dataset management for evaluations",
            "Annotation tools for human feedback"
        ]
        
        for feature in dashboard_features:
            print(f"   {feature}")
        
        print(f"\nAccess your dashboard at: https://smith.langchain.com/")
        print(f"   Your traces will appear in the '{project_name}' project")
        
    except Exception as e:
        print(f"Could not connect to LangSmith: {e}")
        print("Make sure your API key is correct and you have internet access.")
    
    print()


def main():
    """
    Main function to run all demos
    """
    print("LangChain with Amazon Bedrock Examples")
    print("="*50)
    print()
    
    # First show the visual diagrams
    display_langchain_flow_diagrams()
    
    print("\n" + "="*50)
    print("Now let's see these concepts in action!")
    print("="*50)
    
    # Run all demos
    try:
        demo_1_basic_llm()
        time.sleep(2)  # Small delay to avoid throttling
        
        demo_2_prompt_templates()
        time.sleep(2)
        
        demo_3_chains()
        time.sleep(2)
        
        demo_4_conversation_memory()
        time.sleep(2)
        
        demo_5_custom_chain()
        time.sleep(2)
        
        demo_6_few_shot_prompting()
        time.sleep(2)
        
        demo_7_structured_output()

        demo_8_flow_visualization()
        time.sleep(2)
        
        # LangSmith demos
        demo_9_langsmith_tracing()
        time.sleep(2)
        
        demo_10_langsmith_features()
        
        print("="*50)
        print("All demonstrations completed successfully!")
        print("\nKey LangChain Concepts Demonstrated:")
        print("1. Basic LLM initialization and usage")
        print("2. Prompt templates for reusable prompts")
        print("3. Chains for combining operations")
        print("4. Live flow visualization")
        print("5. Memory for maintaining conversation context")
        print("6. Custom chains with multiple processing steps")
        print("7. Few-shot prompting with examples")
        print("8. Structured output for data extraction")
        print("9. LangSmith tracing and monitoring")
        print("10. Advanced LangSmith features")
        
        print("\nVisual Flow Diagrams Shown:")
        print("• Basic LLM Flow")
        print("• Prompt Template Flow") 
        print("• Chain Flow (Sequential)")
        print("• Memory Flow (Context)")
        print("• RAG Flow (Document Retrieval)")
        print("• Agent Flow (Tool Usage)")
        print("• Complete Pipeline Architecture")
        print("• Component Interaction Map")
        
    except Exception as e:
        print(f"Error running demos: {e}")
        print("Make sure your AWS credentials are configured correctly.")


if __name__ == "__main__":
    main()
