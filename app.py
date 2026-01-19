import os
import sys

# Add project directory to sys.path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from rag import get_rag_chain
except ImportError:
    # Handle case where script is run from inside 'project' dir
    sys.path.append(os.path.dirname(__file__))
    from rag import get_rag_chain

def main():
    print("Initializing Swiggy Annual Report RAG System...")
    try:
        qa_chain = get_rag_chain()
    except Exception as e:
        print(f"Error initializing system: {e}")
        print("Please ensure you have run 'python project/ingest.py' and set your GOOGLE_API_KEY.")
        return

    print("\n✅ System Ready! Ask questions about the Swiggy Annual Report.")
    print("Type 'exit', 'quit', or 'q' to end the session.\n")

    while True:
        query = input("❓ Your Question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye! 👋")
            break
        
        if not query:
            continue

        try:
            print("Thinking... ⏳")
            result = qa_chain.invoke(query)
            if isinstance(result, dict):
                answer = result['answer']
                source_docs = result["source_documents"]
            else:
                answer = result
                source_docs = []

            print(f"\n💡 Answer:\n{answer}\n")
            
            # Optional: Display source context
            print("🔍 Sources:")
            for i, doc in enumerate(source_docs):
                print(f"[{i+1}] Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:150]}...")
            
            print("-" * 50)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
