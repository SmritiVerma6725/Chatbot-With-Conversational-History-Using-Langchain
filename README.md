# Chatbot-With-Conversational-History-Using-Langchain

LLM-Powered Chatbot with Memory (LangChain + Groq)
This project demonstrates how to build an intelligent chatbot using LangChain and Groq’s LLM (Gemma2-9b-It). Unlike basic chatbots, this implementation supports conversational memory, allowing it to maintain context across messages and simulate more natural interactions.

Key Features
1.Conversational Memory
Uses RunnableWithMessageHistory to store and recall user inputs across turns, enabling the chatbot to answer follow-up questions contextually.

2.Prompt Engineering with Templates
Implements ChatPromptTemplate to format prompts and include system-level instructions, helping guide the chatbot’s behavior.

3.Multilingual Response Capability
Allows users to interact in various languages by passing language preferences dynamically through prompt variables.

4.Session-Based Chat Tracking
Stores chat history separately for each session using session IDs—ideal for managing multiple user conversations.

5.Message Trimming for Context Management
Uses trim_messages to control how much of the conversation history is passed to the model, preventing overflow of the context window.

6.Step-by-Step Implementation
The notebook walks through the entire process—from setting up the model to building memory-aware chains and managing conversation history.

Ideal Use Cases
1.Personal AI Assistants
2.Customer Support Bots
3.Prototypes for Memory-Enabled RAG Systems
4.Conversational Interfaces for LLM Apps
5.Education & Learning Projects in AI/LLM Development

Built With
1.Python
2.LangChain
3.Groq API (Gemma2-9b-It)
4.dotenv
5.Jupyter Notebook
