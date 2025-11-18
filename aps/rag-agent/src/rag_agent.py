

import os
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, DirectoryLoader


class RAGAgent:
    
    
    def __init__(
        self,
        documents_path: str,
        model_name: str = "gemini-pro",
        temperature: float = 0.3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 4,
        use_local_embeddings: bool = False
    ):
       
        
        load_dotenv()
        
        self.documents_path = documents_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        
       
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY não encontrada! "
                "Crie um arquivo .env com sua chave do Google Gemini"
            )
        
       
        if not model_name.startswith("models/"):
            model_name = f"models/{model_name}"
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=google_api_key,
            convert_system_message_to_human=True
        )
        
      
        if use_local_embeddings:
            print("Usando embeddings locais (HuggingFace)...")
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            print("Usando Google embeddings...")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_api_key
            )
        
       
        self.vectorstore = None
        self.qa_chain = None
        
        print(f"RAG inicializado: {model_name}")
    
    def load_documents(self) -> List[Any]:
       
        print(f"\nCarregando documentos de: {self.documents_path}")
        
        
        loader = DirectoryLoader(
            self.documents_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        
        documents = loader.load()
        print(f"✓ {len(documents)} documento(s) carregado(s)")
        
        return documents
    
    def split_documents(self, documents: List[Any]) -> List[Any]:
        
        print(f"\nDividindo documentos em chunks...")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"✓ {len(chunks)} chunks criados")
        
        return chunks
    
    def create_vectorstore(self, chunks: List[Any]) -> FAISS:
        
        print(f"\nCriando embeddings e vector store...")
        
        
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
       
        vectorstore.save_local("./faiss_index")
        print(f"Vector store criado e salvo em: ./faiss_index")
        
        return vectorstore
    
    def setup(self):
        
        
        print("CONFIGURANDO AGENTE RAG")
        
        
        
        documents = self.load_documents()
        
        if not documents:
            raise ValueError(
                f"Nenhum documento encontrado em {self.documents_path}!\n"
                "Adicione arquivos .txt na pasta 'data/'"
            )
        
        
        chunks = self.split_documents(documents)
        
       
        self.vectorstore = self.create_vectorstore(chunks)
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
      
        template = """
Você é um assistente especializado em Python e desenvolvimento de software.
Use o seguinte contexto para responder à pergunta do usuário de forma clara e precisa.

Se a resposta não estiver no contexto fornecido, diga honestamente que não sabe.
Não invente informações. Seja direto e objetivo.

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA DETALHADA:
"""
        
        PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        
        print("AGENTE RAG CONFIGURADO COM SUCESSO!")
       
    
    def ask(self, question: str, show_sources: bool = True) -> Dict[str, Any]:
        """
        Faz uma pergunta ao agente RAG.
        
        Args:
            question: Pergunta do usuário
            show_sources: Se deve mostrar os documentos fonte
            
        Returns:
            Dicionário com resposta e metadados
        """
        if not self.qa_chain:
            raise ValueError(
                "Agente não configurado! Execute setup() primeiro."
            )
        
        print(f"\nPergunta: {question}")
        
        
        
        result = self.qa_chain({"query": question})
        
        response = {
            "question": question,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
        
        
        print(f"\nResposta:\n{result['result']}")
        
        
        if show_sources and result["source_documents"]:
            print(f"\nDocumentos consultados:")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Desconhecido")
                preview = doc.page_content[:100].replace("\n", " ")
                print(f"{i}. {source}")
                print(f"   Preview: {preview}...")
        
    
        
        return response
    
    def chat(self):
        
        
        
        print("CHAT COM AGENTE RAG")
       
        print("\nDigite suas perguntas (ou 'sair' para encerrar)\n")
        
        while True:
            try:
                question = input("Você: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ["sair", "exit", "quit", "q"]:
                    print("\nEncerrando chat.")
                    break
                
                self.ask(question)
                print()
                
            except KeyboardInterrupt:
                print("\n\nEncerrando chat.")
                break
            except Exception as e:
                print(f"\nErro: {e}\n")


def main():
    
    
    data_path = Path(__file__).parent.parent / "data"
    
    agent = RAGAgent(
        documents_path=str(data_path),
        model_name="gemini-pro",
        temperature=0.3,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=4
    )
    
    
    agent.setup()
    
   
    print("EXEMPLOS DE PERGUNTAS")
    
    
    exemplos = [
        "O que é PEP 8?",
        "Como funcionam os decoradores em Python?",
        "O que é RAG no LangChain?",
        "Quais são as boas práticas para performance em Python?"
    ]
    
    for pergunta in exemplos:
        agent.ask(pergunta, show_sources=False)
        print()
    
    
    agent.chat()


if __name__ == "__main__":
    main()
