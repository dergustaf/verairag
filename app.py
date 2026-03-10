import streamlit as st
import openai
import re
from pinecone import Pinecone
import datetime

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="AI Koučovací Archiv", page_icon="🧠", layout="wide")

# --- PŘIPOJENÍ K DATABÁZI A AI (CACHED) ---
@st.cache_resource
def init_connections():
    # Klíče se načítají ze Streamlit Secrets pro maximální bezpečnost
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("veraibot")
    return client, index

def clean_html(text):
    """Odstraní HTML značky z textu pro čistší kontext."""
    if not text:
        return ""
    return re.sub('<[^<]+?>', '', text)

# --- HLAVNÍ UŽIVATELSKÉ ROZHRANÍ ---
st.title("🧠 AI Koučovací Asistent")
st.markdown("Prohledávejte svůj osobní archiv (podcasty, blogy, newslettery a knihy) pomocí umělé inteligence.")

try:
    client, index = init_connections()
    namespaces = ['__default__']

    # Vstup od uživatele
    query = st.text_input("Na co se chcete kouče zeptat?", placeholder="Např. Jak efektivně zvládat stres?")
    
    # NOVÉ: Možnost filtrovat podle typu a názvu (Variables defined here!)
    col1, col2 = st.columns(2)
    with col1:
        doc_type_filter = st.text_input("Filtrovat podle typu (volitelné):", placeholder="Např. podcast, blog")
    with col2:
        doc_title_filter = st.text_input("Filtrovat podle názvu (volitelné):", placeholder="Např. Čtyřicítka - čas, kdy se TO láme")

    if st.button("Zeptej se kouče") and query:
        with st.spinner("🔍 Prohledávám archivy a připravuji odpověď..."):
            # 1. HyDE Krok (Hypotetický dokument)
            hyde_prompt = f"Napiš krátký odstavec (3 věty) v češtině, jak by mohl vypadat přepis z koučování na téma: '{query}'."
            hyde_res = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": hyde_prompt}])
            hyde_text = hyde_res.choices[0].message.content
            
            # 2. Vektorizace hypotetického textu
            query_vector = client.embeddings.create(input=[hyde_text], model="text-embedding-3-small").data[0].embedding

            # 3. Příprava filtru metadat
            filters = []
            if doc_type_filter:
                filters.append({"type": {"$eq": doc_type_filter}})
            if doc_title_filter:
                filters.append({"title": {"$eq": doc_title_filter}})
                
            metadata_filter = {}
            if len(filters) == 1:
                metadata_filter = filters[0]
            elif len(filters) > 1:
                metadata_filter = {"$and": filters}

            # 4. Vyhledávání v Pinecone (Zvýšeno top_k na 15)
            all_matches = []
            for ns in namespaces:
                results = index.query(
                    vector=query_vector, 
                    top_k=15, 
                    namespace=ns, 
                    include_metadata=True,
                    filter=metadata_filter if metadata_filter else None
                )
                all_matches.extend(results.get('matches', []))

            # Seřazení podle nejlepší shody a oříznutí na top 5
            all_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)[:15]
            
            context_parts = []
            sources_info = []
            
            for m in all_matches:
                meta = m.get('metadata', {})
                raw_text = meta.get('text', meta.get('content_html', ''))
                text = clean_html(raw_text)
                
                # Dynamické určení zdroje a typu z metadat
                title = meta.get('title', meta.get('session_id', meta.get('file_name', 'Archivní záznam')))
                doc_type = meta.get('type', 'Neznámý typ')
                
                # Prvních 100 slov z obsahu
                words = text.split()
                snippet = " ".join(words[:100])
                if len(words) > 100:
                    snippet += "..."
                
                context_parts.append(f"ZDROJ [{title}]: {text}")
                
                # Formátované informace pro zobrazení ve zdrojích
                source_display = (
                    f"**Název:** {title}\n\n"
                    f"**Typ:** {doc_type}\n\n"
                    f"**Míra shody:** {m['score']*100:.1f}%\n\n"
                    f"**Obsah (prvních 100 slov):** {snippet}\n\n"
                    f"---"
                )
                sources_info.append(source_display)

            # 5. Finální syntéza v češtině
            if not all_matches:
                st.warning("Nebyl nalezen žádný odpovídající obsah pro tento dotaz a filtr.")
            else:
                final_res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Jsi profesionální koučovací asistent. Shrň rady z poskytnutých českých úryvků do jasné a srozumitelné odpovědi v češtině. Cituj zdroje jako [Název zdroje]."},
                        {"role": "user", "content": f"Kontext: {' '.join(context_parts)}\n\nOtázka: {query}"}
                    ]
                )
                
                odpoved = final_res.choices[0].message.content
                
                # --- ZOBRAZENÍ VÝSLEDKŮ ---
                st.subheader("🟢 Rada kouče")
                st.write(odpoved)
                
                with st.expander("Zobrazit použité zdroje"):
                    for info in sources_info:
                        st.markdown(info)

                # --- TLAČÍTKO PRO STAŽENÍ ---
                export_report = f"DOTAZ: {query}\nDATUM: {datetime.datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n{odpoved}\n\nZDROJE:\n" + "\n".join(sources_info)
                st.download_button(
                    label="Stáhnout shrnutí (.txt)",
                    data=export_report.encode('utf-8'),
                    file_name=f"rada_kouce_{datetime.date.today()}.txt",
                    mime="text/plain"
                )

except Exception as e:
    st.error(f"Chyba konfigurace: Ujistěte se, že máte nastaveny API klíče v Streamlit Secrets. Detaily: {e}")

