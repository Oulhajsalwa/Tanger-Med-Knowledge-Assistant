# ⚓ Tanger Med Knowledge Assistant

> Système **RAG multi-années** pour l'exploration et l'analyse intelligente des
> Rapports Annuels et Rapports RSE du Groupe Tanger Med.
> Réponses générées par GPT-4o, **strictement sourcées**, avec citations
> page par page et **abstention** lorsque l'information n'existe pas dans le corpus.

---

## 1. Présentation

Ce projet est un prototype professionnel de **Retrieval-Augmented Generation**
appliqué au corpus documentaire public de Tanger Med. Il permet d'interroger en
langage naturel plusieurs années de rapports (annuels et RSE), de comparer des
indicateurs entre exercices, et d'obtenir systématiquement la trace exacte
(document, année, page) de chaque information utilisée.

L'accent est mis sur les points qui font la différence entre un « chatbot sur
PDF » et un vrai système RAG d'entreprise :

| Enjeu | Traitement dans ce projet |
|---|---|
| Traçabilité | Citations document + année + **page**, extrait source consultable |
| Hallucinations | Double garde-fou d'abstention + prompt strict + test de fidélité numérique |
| Questions multi-années | Retrieval **équilibré par année**, jamais un top-k global |
| Chiffres / acronymes | Recherche **hybride** (vectorielle + BM25), pas seulement sémantique |
| Corpus réel désordonné | Nettoyage des en-têtes répétés, déduplication, correction de documents mal classés |
| Évaluation | Harnais de mesure dédié (retrieval, abstention, citations, fidélité) |

---

## 2. Problématique

Les rapports Tanger Med représentent plusieurs centaines de pages par année,
réparties sur de multiples documents (annuel + RSE) et exercices. La consultation
manuelle rend difficile :

- la recherche d'une information précise dans des documents volumineux ;
- la comparaison d'un indicateur entre plusieurs exercices ;
- l'identification de l'évolution d'une trajectoire (trafic, décarbonation, RH) ;
- la **traçabilité** de la réponse jusqu'à sa page d'origine ;
- la synthèse d'informations dispersées dans plusieurs documents.

## 3. Objectifs

1. Répondre en langage naturel à partir d'un corpus multi-années.
2. Citer précisément les sources (document, année, page, section).
3. Gérer le raisonnement temporel (année unique, comparaison, évolution).
4. **Ne jamais inventer** : refuser explicitement quand l'information est absente.
5. Exposer le tout via une API propre et une interface de démonstration.
6. Mesurer objectivement la qualité du système.

---

## 4. Architecture

```mermaid
flowchart TD
    subgraph IN[" INGESTION "]
        A[PDF Rapports Annuels / RSE] --> B[Extraction PyMuPDF<br/>page par page]
        B --> C[Nettoyage<br/>en-têtes/pieds répétés]
        C --> D[Chunking structurel<br/>sections + pages + overlap]
        D --> E[Enrichissement métadonnées<br/>année, type, page, section]
        E --> F[Embeddings<br/>text-embedding-3-large]
        F --> G[(ChromaDB<br/>vecteurs + metadata)]
        E --> H[(Index BM25<br/>persisté)]
    end

    subgraph Q["REQUÊTE"]
        U[Question utilisateur] --> QA[Analyse déterministe<br/>années · intention · thème · type]
        QA --> MF[Filtrage metadata<br/>year / document_type]
        MF --> VS[Recherche vectorielle]
        MF --> BM[Recherche BM25]
        G -.-> VS
        H -.-> BM
        VS --> FU[Fusion hybride<br/>α·sem + 1-α·lex]
        BM --> FU
        FU --> AB{Seuils<br/>d'abstention}
        AB -- sous le seuil --> NA[Réponse d'abstention<br/>aucun appel LLM]
        AB -- au-dessus --> GEN[GPT-4o<br/>prompt strict sourcé]
        GEN --> OUT[Réponse + Citations]
    end

    OUT --> UI[Streamlit]
    OUT --> API[FastAPI]
```

### Flux multi-années

Pour une question de comparaison ou d'évolution, le système **n'exécute pas** une
recherche globale : il lance une recherche filtrée **par année**, garantissant que
chaque exercice demandé dispose de sa part du contexte.

```
"Comparez le CA entre 2024 et 2025"
   → recherche(years=[2024]) → k chunks de 2024
   → recherche(years=[2025]) → k chunks de 2025
   → contexte équilibré → GPT-4o
```

---

## 5. Technologies

| Couche | Choix | Justification |
|---|---|---|
| LLM | **OpenRouter** → `openai/gpt-4o` | API compatible OpenAI, modèle imposé par le cahier des charges |
| Embeddings | `openai/text-embedding-3-large` (3072d) | Qualité multilingue (corpus français) |
| Extraction PDF | **PyMuPDF** | Rapide, fiable, conserve le numéro de page |
| Vector DB | **ChromaDB** (persistant) | Embarqué, sans serveur, index HNSW, filtrage metadata natif |
| Lexical | **rank-bm25** | Complément indispensable pour chiffres/acronymes/années |
| API | **FastAPI** + Uvicorn | Typage Pydantic, OpenAPI auto |
| UI | **Streamlit** | Démonstration rapide et lisible |
| Fiabilité | **tenacity** | Retries bornés avec backoff exponentiel |

---

## 6. Structure du projet

```
tanger-med-rag/
├── app/
│   ├── api/              # FastAPI : routes, schémas, application
│   ├── config/           # Configuration centralisée (pydantic-settings)
│   ├── ingestion/        # loader · cleaner · chunker · pipeline · CLI
│   ├── retrieval/        # vector_store · bm25 · hybrid
│   ├── rag/              # query_analyzer · prompts · generator · pipeline
│   ├── models/           # Schémas Pydantic partagés
│   └── utils/            # logging · text · openrouter_client
├── data/
│   ├── annual_reports/   # Rapports annuels (PDF)
│   └── rse_reports/      # Rapports RSE (PDF)
├── evaluation/           # questions.json + evaluate.py
├── scripts/              # check_openrouter.py 
├── vectorstore/          # ChromaDB + index BM25 
├── streamlit_app.py
├── requirements.txt
└── .env.example
```

---

## 7. Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 8. Configuration OpenRouter

Copiez `.env.example` vers `.env` puis renseignez votre clé :

```env
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=openai/gpt-4o
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-large
OPENROUTER_EMBEDDING_DIM=3072
```




```bash
python scripts/check_openrouter.py
```



## 9. Ajout des rapports

Déposez les PDF officiels ([tangermed.ma/fr/documentation](https://www.tangermed.ma/fr/documentation/),
thématique « Rapports Annuels et RSE ») dans :

```
data/annual_reports/   →  rapport_annuel_2024.pdf, ...
data/rse_reports/      →  rapport_rse_2024.pdf, ...
```


Le corpus actuellement indexé — les six PDF officiels complets, **2 526 chunks** :

| Document | Pages | Chunks |
|---|---|---|
| Rapport Annuel 2023 | 119 | 313 |
| Rapport Annuel 2024 | 130 | 555 |
| Rapport Annuel 2025 | 109 | 490 |
| Rapport RSE 2023 | 101 | 324 |
| Rapport RSE 2024 | 99 | 391 |
| Rapport RSE 2025 | 116 | 453 |



## 11. Lancement du backend

```bash
uvicorn app.api.app:app --reload --port 8000
```

| Endpoint | Description |
|---|---|
| `POST /api/query` | Question → réponse + citations |
| `GET /api/health` | État de l'index et de la configuration |
| `GET /api/documents` | Inventaire du corpus indexé |
| `GET /docs` | Documentation OpenAPI interactive |

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Quel a été le tonnage global en 2025 ?","years":[2025]}'
```

## 12. Lancement de l'interface

```bash
streamlit run streamlit_app.py
```




---

## 13. Exemples de questions

**Factuelles**
- Quel a été le tonnage global de marchandises traité en 2025 ?
- Quel est le chiffre d'affaires consolidé du Groupe et sa progression ?
- Quelle est la part de femmes managers au sein du Groupe ?

**Thématiques**
- Quelles sont les principales actions de décarbonation menées par Tanger Med ?
- Comment est organisée la gouvernance RSE du Groupe ?
- Quels dispositifs de sécurité des systèmes d'information sont en place ?

**Temporelles / comparatives**
- Comparez le chiffre d'affaires entre 2024 et 2025.
- Comment le trafic du complexe portuaire a-t-il évolué entre 2023 et 2025 ?
- Comparez la part de femmes managers entre 2023 et 2025.

**Hors corpus (doivent déclencher l'abstention)**
- Quel était le chiffre d'affaires de Tanger Med en 2030 ?
- Quel est le cours de l'action Apple aujourd'hui ?

---

## 14. Architecture RAG en détail

### Chunking structurel
Le découpage n'est pas un simple `split` de N caractères :
- détection heuristique des titres (casse, longueur, ponctuation, densité de chiffres) ;
- suivi de la **section courante** reportée dans les métadonnées de chaque chunk ;
- suivi de la **plage de pages** (`page_start` / `page_end`) — un chunk peut
  chevaucher deux pages, la citation reste exacte (`p. 90-91`) ;
- overlap configurable pour ne pas perdre le contexte aux frontières ;
- repli phrase-à-phrase pour les paragraphes plus longs qu'un chunk.

### Recherche hybride
```
score = RETRIEVAL_ALPHA × score_sémantique + (1 − RETRIEVAL_ALPHA) × score_lexical
```
Les deux composantes sont normalisées dans `[0,1]` :
- **sémantique** : similarité cosinus brute (non rééchelonnée, pour rester interprétable) ;
- **lexical** : transformation saturante `raw / (raw + k)`.

> **Décision de conception importante.** Une normalisation min-max ou max
> attribue toujours `1.0` au meilleur résultat, *même pour une question sans
> rapport avec le corpus* — ce qui rend tout seuil d'abstention inopérant. La
> transformation saturante conserve l'information de magnitude absolue.


---

## 15. Stratégie anti-hallucination

Trois lignes de défense **indépendantes** :

**1. Filtrage structurel** — une question sur 2030 alors que le corpus s'arrête
en 2025 ne retourne aucun chunk : l'abstention est mécanique.

**2. Double seuil d'abstention** (avant tout appel LLM, donc sans coût) :

| Garde-fou | Variable | Rôle |
|---|---|---|
| Score fusionné | `RELEVANCE_THRESHOLD` | Rien de suffisamment pertinent |
| Score sémantique | `MIN_SEMANTIC_SCORE` | Question **hors sujet** |

Le second garde-fou est essentiel et découle d'une mesure réelle sur ce corpus :
la question *« Quel est le cours de l'action Apple ? »* obtient un score BM25 de
**13,3** — supérieur à une question environnementale légitime (**9,1**) — parce
que « action » et « cours » saturent le vocabulaire des rapports. **Seul le score
sémantique distingue le hors-sujet.**

**3. Prompt système strict** — interdiction de répondre hors contexte, de mélanger
les années, de recalculer un chiffre, obligation de citer et de signaler
explicitement toute information manquante.


## 17. Évaluation

```bash
python evaluation/evaluate.py
python evaluation/evaluate.py --output evaluation/results.json
```

19 questions réparties en 6 catégories (factuelles, thématiques, temporelles,
comparaisons, évolutions, hors corpus). Métriques :

| Métrique | Mesure |
|---|---|
| `retrieval_hit` | Un passage récupéré contient l'information attendue |
| `year_coverage` | **Chaque** année demandée est représentée dans les citations |
| `abstention_correct` | Le système s'abstient exactement quand il le doit |
| `citation_valid` | Citations complètes et années réellement présentes au corpus |
| `faithfulness_numeric` | **Tout nombre écrit dans la réponse existe dans le contexte** |
| `answer_keywords` | Les éléments attendus figurent dans la réponse |

`faithfulness_numeric` est un contrôle déterministe et peu coûteux,
particulièrement adapté à un corpus dense en indicateurs : il détecte les KPI
inventés, la forme d'hallucination la plus dommageable ici.

**Dernier relevé** avec `openai/gpt-4o` + `openai/text-embedding-3-large`, sur un
corpus de 1 647 chunks (les rapports 2025 n'étaient alors indexés que
partiellement) :

```
Abstention correcte  : 19/19 (100%)
Couverture des années: 11/11 (100%)
Citations valides    : 19/19 (100%)
Retrieval hit        : 15/15 (100%)
Mots-clés attendus   : 10/10 (100%)
Fidélité numérique   : 11/15  (73%)
```







## Récapitulatif des commandes

```bash
pip install -r requirements.txt        # installation
python scripts/check_openrouter.py     # vérifier la clé et les modèles
python -m app.ingestion --force        # indexer les rapports
streamlit run streamlit_app.py         # interface (http://localhost:8501)

python evaluation/evaluate.py          # évaluation
```
