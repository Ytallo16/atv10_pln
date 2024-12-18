import PyPDF2
import re
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Função para limpar o texto
def clean_text(text):
    # Remove caracteres especiais, múltiplos espaços e converte para minúsculas
    text = re.sub(r'\s+', ' ', text)  # Remove múltiplos espaços
    text = re.sub(r'[^\w\s]', '', text)  # Remove pontuações
    text = text.lower()  # Converte para minúsculas
    return text

# Função para gerar uma nuvem de palavras
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Função para contar palavras mais frequentes
def get_most_frequent_words(text, num_words=10):
    vectorizer = CountVectorizer(stop_words='english', max_features=num_words)
    word_counts = vectorizer.fit_transform([text])
    word_freq = dict(zip(vectorizer.get_feature_names_out(), word_counts.toarray()[0]))
    return word_freq

# Programa principal
if __name__ == "__main__":
    pdf_path = "arquivo_combinado.pdf"
    
    try:
        # Extração de texto
        print("Extraindo texto do PDF...")
        raw_text = extract_text_from_pdf(pdf_path)
        print("Texto extraído com sucesso!")

        # Limpeza do texto
        print("Limpando o texto...")
        cleaned_text = clean_text(raw_text)
        print("Texto limpo com sucesso!")

        # Palavras mais frequentes
        print("Analisando palavras mais frequentes...")
        frequent_words = get_most_frequent_words(cleaned_text)
        print("Palavras mais frequentes:")
        for word, count in frequent_words.items():
            print(f"{word}: {count}")

        # Geração de nuvem de palavras
        print("Gerando nuvem de palavras...")
        generate_wordcloud(cleaned_text)

    except FileNotFoundError:
        print("Erro: Arquivo não encontrado. Verifique se 'arquivo_combinado.pdf' está no mesmo diretório do script.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
