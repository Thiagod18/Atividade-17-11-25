

from pathlib import Path
from rag_agent import RAGAgent


def main():
    
    data_path = Path(__file__).parent.parent / "data"
    
    
    print("Criando agente RAG...")
    agent = RAGAgent(
        documents_path=str(data_path),
        model_name="gemini-2.0-flash",  
        temperature=0.3,                
        chunk_size=1000,                
        chunk_overlap=200,              
        top_k=4,                        
        use_local_embeddings=True      
    )
    
    
    agent.setup()
    
    
    
    print("EXEMPLOS DE USO")
   
    
    
    agent.ask("O que são decoradores em Python?")
    
    
    agent.ask("Como funciona o RAG no LangChain?", show_sources=False)
    
    
    
    print("Iniciando modo chat interativo...")
    
    agent.chat()


if __name__ == "__main__":
    main()
