import PyPDF2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Embedding, Dropout

# Função para extrair texto de um arquivo PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Função para gerar o vetor TF-IDF
def generate_tfidf_vectors(text):
    vectorizer = TfidfVectorizer(max_features=500)  # Limita a 500 características
    X = vectorizer.fit_transform([text]).toarray()
    return X, vectorizer

# Função para criar rótulos fictícios (exemplo: 0 para análise positiva, 1 para negativa)
def generate_fake_labels(size):
    return np.random.randint(0, 2, size=size)

# Programa principal
if __name__ == "__main__":
    pdf_path = "texto_processado.pdf"
    try:
        # Extração do texto
        print("Extraindo texto do PDF...")
        raw_text = extract_text_from_pdf(pdf_path)
        print("Texto extraído com sucesso!")

        # Transformação TF-IDF
        print("Gerando vetores TF-IDF...")
        X, vectorizer = generate_tfidf_vectors(raw_text)
        y = generate_fake_labels(X.shape[0])
        print("Vetores TF-IDF gerados com sucesso!")

        # Divisão dos dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Construção do modelo RNN
        print("Construindo modelo RNN...")
        model = Sequential([
            Dense(128, input_dim=X_train.shape[1], activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        # Compilação do modelo
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("Modelo RNN construído com sucesso!")

        # Treinamento do modelo
        print("Treinando modelo...")
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Avaliação do modelo
        print("Avaliando modelo...")
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"Acurácia do modelo: {accuracy * 100:.2f}%")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{pdf_path}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
