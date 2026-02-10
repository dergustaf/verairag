import streamlit as st
import openai
import re
from pinecone import Pinecone
import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Coaching Archive", page_icon="🧠", layout="wide")

# --- DATABASE & AI SETUP (CACHED) ---
@st.cache_resource
def init_connections():
    # We pull keys from Streamlit Secrets for security
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("veraibot1536")
    return client, index

def clean_html(text):
    return re.sub('<[^<]+?>', '', text) if text else ""

# --- MAIN UI ---
st.title("🧠 AI Coaching Assistant")
st.markdown("Query your personal archives (Podcasts, Blogs, Newsletters, and Books) using AI.")

try:
    client, index = init_connections()
    namespaces = ['newsletter_cs', 'podcast_cs', 'blog-cs', 'book-mybook-cs']

    # User Input
    query = st.text_input("What would you like to ask the coach?", placeholder="e.g., How to handle stress?")

    if st.button("Ask the Coach") and query:
        with st.spinner("🔍 Searching archives and synthesizing advice..."):
            # 1. HyDE Step (Hypothetical Document Embedding)
            # We ask the AI to imagine a Czech response to better match the Czech transcripts
            hyde_prompt = f"Napiš krátký odstavec (3 věty) v češtině, jak by mohl vypadat přepis z koučování na téma: '{query}'."
            hyde_res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": hyde_prompt}])
            hyde_text = hyde_res.choices[0].message.content
            
            # 2. Vectorize the HyDE text
            query_vector = client.embeddings.create(input=[hyde_text], model="text-embedding-3-small").data[0].embedding

            # 3. Search Pinecone across all namespaces
            all_matches = []
            for ns in namespaces:
                results = index.query(vector=query_vector, top_k=3, namespace=ns, include_metadata=True)
                all_matches.extend(results['matches'])

            # Sort by best matches
            all_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)[:5]
            
            context_parts = []
            sources_info = []
            for m in all_matches:
                meta = m['metadata']
                text = clean_html(meta.get('content_html', meta.get('text', '')))
                # Better source naming
                source = meta.get('title') or meta.get('session_id') or meta.get('file_name') or "Archive Transcript"
                context_parts.append(f"SOURCE [{source}]: {text}")
                sources_info.append(f"- **{source}** (Confidence: {m['score']*100:.1f}%)")

            # 4. Final English Synthesis
            final_res = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a coaching assistant. Synthesize the advice from the provided Czech snippets into a clear, professional English response. Cite sources as [Source Name]."},
                    {"role": "user", "content": f"Context: {' '.join(context_parts)}\n\nQuestion: {query}"}
                ]
            )
            
            answer = final_res.choices[0].message.content
            
            # --- DISPLAY RESULTS ---
            st.subheader("🟢 Coach's Advice")
            st.write(answer)
            
            with st.expander("View Data Sources"):
                st.markdown("\n".join(sources_info))

            # --- DOWNLOAD BUTTON ---
            full_report = f"QUERY: {query}\nDATE: {datetime.datetime.now()}\n\n{answer}\n\nSOURCES:\n" + "\n".join(sources_info)
            st.download_button(
                label="Download Summary (.txt)",
                data=full_report,
                file_name=f"coach_advice_{datetime.date.today()}.txt",
                mime="text/plain"
            )

except Exception as e:
    st.error(f"Configuration Error: Please ensure your API keys are set in Streamlit Secrets. Error: {e}")