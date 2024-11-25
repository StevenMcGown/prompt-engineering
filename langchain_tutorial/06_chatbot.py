from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import panel as pn
import param
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

class ChatBotFileSystem(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super().__init__(**params)
        self.panels = []
        self.loaded_file = "docs/MachineLearning-Lecture01.pdf"
        self.qa = load_db(self.loaded_file, "stuff", 4)

    def call_load_db(self, count):
        if count == 0 or file_input.value is None:
            return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")
        else:
            file_input.save("temp.pdf")
            self.loaded_file = file_input.filename
            button_load.button_style = "outline"
            self.qa = load_db("temp.pdf", "stuff", 4)
            button_load.button_style = "solid"
        self.clear_history()  # Clear chat history when loading new DB
        return pn.pane.Markdown(f"Loaded File: {self.loaded_file}")

    def conversation_chain(self, query):
        if not query:
            return pn.WidgetBox(pn.Row("User:", pn.pane.Markdown("", width=600)), scroll=True)

        # Prevent duplicate execution
        if self.chat_history and self.chat_history[-1][0] == query:
            print(f"Duplicate query ignored: {query}")
            return pn.WidgetBox(*self.panels, scroll=True)

        # Perform the query
        try:
            result = self.qa.invoke({"question": query, "chat_history": self.chat_history})

            # Append query and response to chat history
            self.chat_history.append((query, result["answer"]))
            print("Updated Chat History:", self.chat_history)

            # Update response variables
            self.db_query = result.get("generated_question", "")
            self.db_response = result.get("source_documents", [])
            self.answer = result.get("answer", "")

            # Add new panels for the chat
            self.panels.extend([
                pn.Row("User:", pn.pane.Markdown(query, width=600)),
                pn.Row("ChatBot:", pn.pane.Markdown(f"<div style='background-color:#F6F6F6;'>{self.answer}</div>", width=600)),
            ])
        except Exception as e:
            print(f"Error during conversation chain: {e}")
            self.panels.append(pn.Row("Error:", pn.pane.Markdown("Something went wrong.", width=600)))

        inp.value = ""  # Clear input field
        return pn.WidgetBox(*self.panels, scroll=True)

    def clear_history(self, count=0):
        self.chat_history = []
        self.panels = []  # Clear the panels as well
        print("Chat history cleared.")

def load_db(file, chain_type, k):
    # Load documents from the file
    loader = PyPDFLoader(file)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)

    # Define embedding model
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create vector database
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)

    # Define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})

    # Create chatbot chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa

# Initialize the chatbot
cb = ChatBotFileSystem()

# Panel UI elements
file_input = pn.widgets.FileInput(accept=".pdf")
button_load = pn.widgets.Button(name="Load DB", button_type="primary")
button_clearhistory = pn.widgets.Button(name="Clear History", button_type="warning")
button_clearhistory.on_click(cb.clear_history)
inp = pn.widgets.TextInput(placeholder="Enter text here…")

bound_button_load = pn.bind(cb.call_load_db, button_load.param.clicks)
conversation = pn.bind(cb.conversation_chain, inp)

jpg_pane = pn.pane.Image("./img/convchain.jpg")

# Tabs for the UI
tab1 = pn.Column(
    pn.Row(inp),
    pn.layout.Divider(),
    pn.panel(conversation, loading_indicator=True, height=300),
    pn.layout.Divider(),
)
tab4 = pn.Column(
    pn.Row(file_input, button_load, bound_button_load),
    pn.Row(button_clearhistory, pn.pane.Markdown("Clears chat history. Can use to start a new topic")),
    pn.layout.Divider(),
    pn.Row(jpg_pane.clone(width=400)),
)

dashboard = pn.Column(
    pn.Row(pn.pane.Markdown("# ChatWithYourData_Bot")),
    pn.Tabs(("Conversation", tab1), ("Configure", tab4)),
)

# Display the dashboard
dashboard.show()
