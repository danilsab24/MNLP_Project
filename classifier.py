import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE, trustworthiness
from sklearn.svm import SVC

# define stopwords and tokenizer
default_stopwords = set([
    'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'if', 'in', 'to',
    'of', 'for', 'by', 'with', 'was', 'as', 'that', 'it', 'this', 'are', 'be',
    'from', 'has', 'have', 'had', 'but', 'they', 'their', 'its', 'not', 'all'
])

# clean and tokenize text
def clean_and_tokenize(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in default_stopwords and len(t) > 2]
    return tokens

# load pre-trained word2vec model
print("loading pre-trained model...")
model_path = "/Users/riccardodaguanno/Downloads/GoogleNews-vectors-negative300.bin.gz"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)
print("model loaded.")

# load dataset
df = pd.read_csv("train_dataset_1_with_text.csv")
df = df.dropna(subset=["name", "description", "label"])

names = df["name"].tolist()
descriptions = df["description"].tolist()
labels = df["label"].tolist()

# build average embeddings + extra features
name_vectors = []
valid_names = []
valid_labels = []

for i, (name, description, label) in enumerate(zip(names, descriptions, labels)):
    # get full text (name + description + text column)
    text = df.loc[i, "text"] if "text" in df.columns else ""
    combined_text = f"{name} {description} {text}"
    tokens = clean_and_tokenize(combined_text)
    token_vectors = [model[t] for t in tokens if t in model]

    if token_vectors:
        # compute average word2vec embedding
        avg_vector = np.mean(token_vectors, axis=0)

        # extract extra numeric features
        name_length = len(name)
        num_words = len(name.split())
        special_char_ratio = sum(1 for c in name if not c.isalnum()) / len(name) if len(name) > 0 else 0
        uppercase_count_ratio = sum(1 for c in name if c.isupper()) / len(name) if len(name) > 0 else 0

        # read optional columns
        try:
            wiki_pages = float(df.loc[i, "wiki_pages"]) if "wiki_pages" in df.columns else 0.0
        except:
            wiki_pages = 0.0
        try:
            keywords_count = float(df.loc[i, "keywords_count"]) if "keywords_count" in df.columns else 0.0
        except:
            keywords_count = 0.0

        # concatenate embedding + features
        extra_features = np.array([
            name_length,
            num_words,
            special_char_ratio,
            uppercase_count_ratio,
            wiki_pages,
            keywords_count
        ])
        final_vector = np.concatenate([avg_vector, extra_features])

        name_vectors.append(final_vector)
        valid_names.append(name)
        valid_labels.append(label)

# classification
if name_vectors:
    name_vectors_np = np.array(name_vectors)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(valid_labels)

    # split train and test
    X_train, X_test, y_train, y_test = train_test_split(name_vectors_np, y, test_size=0.2, random_state=42)

    # normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # define svm classifier
    clf = SVC(
        kernel='rbf',
        C=1,
        gamma='scale',
        class_weight={0: 1, 1: 1, 2: 2},
        probability=True,
        random_state=42
    )

    # train model
    clf.fit(X_train, y_train)

    # evaluate on test set
    y_pred = clf.predict(X_test)

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    print("CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    print(cm)

    # plot confusion matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title("Normalized Confusion Matrix")
    plt.show()

    # dimensionality reduction with t-SNE
    max_perplexity = min(50, len(name_vectors_np) - 1)
    tsne = TSNE(n_components=2, perplexity=max_perplexity, max_iter=1000, random_state=42)
    reduced = tsne.fit_transform(name_vectors_np)
    print("\nt-SNE completed.")

    trust = trustworthiness(name_vectors_np, reduced, n_neighbors=5)
    print(f"trustworthiness: {trust:.4f}")

    # plot t-SNE visualization
    unique_labels = list(label_encoder.classes_)
    label_to_color = {label: plt.cm.get_cmap("tab10")(i % 10) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(14, 10))
    for i, name in enumerate(valid_names):
        color = label_to_color[label_encoder.inverse_transform([y[i]])[0]]
        plt.scatter(reduced[i, 0], reduced[i, 1], color=color, s=25, alpha=0.7)
        if len(valid_names) <= 150:
            plt.annotate(name, (reduced[i, 0], reduced[i, 1]), fontsize=8)

    # add legend
    for label, color in label_to_color.items():
        plt.scatter([], [], color=color, label=label)
    plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title("t-SNE of names (Word2Vec) – Trustworthiness: {:.4f}".format(trust))
    plt.grid(True)
    plt.tight_layout()
    plt.show()

else:
    print("no valid embeddings found. check the data or vocabulary.")

