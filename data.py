import pandas as pd

# Create sample data
data = {
    "source": [
        "Hello, how are you?",
        "This is a beautiful day.",
        "I love machine learning.",
        "Let's go for a walk.",
        "What time is it?"
    ],
    "target_fr": [
        "Bonjour, comment ça va ?",
        "C'est une belle journée.",
        "J'adore l'apprentissage automatique.",
        "Allons nous promener.",
        "Quelle heure est-il ?"
    ],
    "target_de": [
        "Hallo, wie geht es dir?",
        "Das ist ein schöner Tag.",
        "Ich liebe maschinelles Lernen.",
        "Lass uns spazieren gehen.",
        "Wie spät ist es?"
    ],
    "target_es": [
        "Hola, ¿cómo estás?",
        "Este es un hermoso día.",
        "Me encanta el aprendizaje automático.",
        "Vamos a dar un paseo.",
        "¿Qué hora es?"
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("sample.csv", index=False)
